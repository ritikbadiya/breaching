""" Implements the GISMN attacks on ViT models.
This method uses L2 loss optimization for the parameter gradients,
And the positional embedding gradients are also recovered using cosine similarity loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time

from .optimization_based_attack import OptimizationBasedAttacker
from .auxiliaries.objectives import Euclidean, CosineSimilarity, objective_lookup
from .auxiliaries.augmentations import augmentation_lookup
from .auxiliaries.text_embedding import TextEmbedder, encode_classes, find_closest_imagenet_classes
from .auxiliaries.class_names import cifar10_classes, imagenet_classes

from .auxiliaries.BigGAN.model import (
    BigGAN,
    Generator,
    biggan_model_name_for_output_dim,
    resolve_pretrained_biggan_weights,
)
from .auxiliaries.BigGAN.config import BigGANConfig, BigGAN32, BigGAN128, BigGAN256, BigGAN512
from .auxiliaries.BigGAN.utils import truncated_noise_sample, save_as_images, one_hot_from_names, one_hot_from_int

from .auxiliaries.stylegan_xl import legacy, dnnlib
from .auxiliaries.stylegan_xl.torch_utils import misc
from .auxiliaries.stylegan2_ada import legacy as stylegan2_ada_legacy, dnnlib as stylegan2_ada_dnnlib
from .auxiliaries.stylegan2_ada.torch_utils import misc as stylegan2_ada_misc

import logging

log = logging.getLogger(__name__)

def build_stylegan_xl_generator(
    device,
    network_pkl=None,
    class_name='training.networks_stylegan2.Generator',
    init_kwargs=None,
    common_kwargs=None,
):
    """Construct a StyleGAN-XL generator and optionally load generator-only weights from a pickle."""
    common_kwargs = common_kwargs or {}
    if network_pkl is not None:
        with dnnlib.util.open_url(network_pkl) as f:
            data = legacy.load_network_pkl(f)
        src_g = data['G_ema']
        log.info(f"StyleGAN-XL init_kwargs: {src_g.init_kwargs}")
        # Prefer the generator defined in the pickle to avoid class/kwargs mismatches.
        if init_kwargs is None:
            return src_g.to(device).eval()
        # If caller insists on reconstruction, use the pickle's class_name unless overridden.
        init_kwargs = dict(init_kwargs)
        init_kwargs.setdefault('class_name', src_g.init_kwargs.get('class_name', class_name))
        G = dnnlib.util.construct_class_by_name(**init_kwargs).to(device).eval()
        misc.copy_params_and_buffers(src_g, G, require_all=False)
        return G

    if init_kwargs is None:
        init_kwargs = dict(class_name=class_name, **common_kwargs)
    else:
        init_kwargs = dict(init_kwargs)
        init_kwargs.setdefault('class_name', class_name)
        init_kwargs.update(common_kwargs)
    return dnnlib.util.construct_class_by_name(**init_kwargs).to(device).eval()

def build_stylegan2_ada_generator(
    device,
    network_pkl=None,
    class_name='training.networks.Generator',
    init_kwargs=None,
    common_kwargs=None,
):
    """Construct a StyleGAN2-ADA generator and optionally load generator-only weights from a pickle."""
    common_kwargs = common_kwargs or {}
    if network_pkl is not None:
        with stylegan2_ada_dnnlib.util.open_url(network_pkl) as f:
            data = stylegan2_ada_legacy.load_network_pkl(f)
        src_g = data['G_ema']
        if init_kwargs is None:
            return src_g.to(device).eval()
        init_kwargs = dict(init_kwargs)
        init_kwargs.setdefault('class_name', src_g.init_kwargs.get('class_name', class_name))
        G = stylegan2_ada_dnnlib.util.construct_class_by_name(**init_kwargs).to(device).eval()
        stylegan2_ada_misc.copy_params_and_buffers(src_g, G, require_all=False)
        return G

    if init_kwargs is None:
        init_kwargs = dict(class_name=class_name, **common_kwargs)
    else:
        init_kwargs = dict(init_kwargs)
        init_kwargs.setdefault('class_name', class_name)
        init_kwargs.update(common_kwargs)
    return stylegan2_ada_dnnlib.util.construct_class_by_name(**init_kwargs).to(device).eval()

class SemanticSimilarityBasedLatentCodeAttackerPEFT(OptimizationBasedAttacker):
    """Implements an optimization-based attack that only recovers the patch information 
    by opyimization L2 loss on the the parameter gradients.
    And also optimizes the cosine similarity loss on the PosEmbed gradients
    """
    def __init__(self, model, loss_fn, cfg_attack, setup=dict(dtype=torch.float, device=torch.device("cpu"))):
        super().__init__(model, loss_fn, cfg_attack, setup)
        objective_fn = objective_lookup.get(self.cfg.objective.type)
        if objective_fn is None:
            raise ValueError(f"Unknown objective type {self.cfg.objective.type} given.")
        else:
            self.objective = objective_fn(scale=self.cfg.objective.scale, 
                                        task_regularization=self.cfg.objective.task_regularization,
                                        posembed_scale=self.cfg.objective.posembed_scale)
            
        self.truncation = self.cfg.objective.truncation if hasattr(self.cfg.objective, 'truncation') else 0.4
        self._biggan_weights_path = None
        self._biggan_state_dict = None

        self.text_embedder = TextEmbedder(model_name="clip", device=self.setup["device"])

        self.cifar_classnames = cifar10_classes
        self.imagenet_classnames = imagenet_classes

        self.templates = [
            "a photo of a {}",
            "an image of a {}",
            "a picture of a {}",
            # "a blurry photo of a {}",
            # "a black and white photo of a {}",
            "a cropped photo of a {}",
            "a close-up photo of a {}",
            "a bright photo of a {}",
        ]

        if self.cfg.generator.nclasses==1000:
            self.imagenet = True
            self.num_classes = 1000
            log.info("Using ImageNet classnames for conditional generation since num_classes is 1000.")
        else:
            self.imagenet = False
        
        if self.cfg.generator.type == "biggan":
            if not self.imagenet:
                self.bigganconfig = BigGAN32
            else:
                self.bigganconfig = BigGAN128
            self.bigganconfig.num_classes = self.num_classes

    def _infer_cifarclassname_from_label(self, label):
        return self.cifar_classnames[label]

    def _compute_imagenet_class_index(self, cifar_labels, imagenet_classnames, return_indices=False):
        # This is a simple heuristic mapping based on substring matching. It can be improved with better NLP techniques if needed.
        cifar_classnames = [self._infer_cifarclassname_from_label(label) for label in cifar_labels]
        log.info(f"Mapping CIFAR classnames {cifar_classnames} to ImageNet classes for conditional generation.")
        cifar_embeds = encode_classes(cifar_classnames, templates=self.templates, embedder=self.text_embedder)
        imagenet_embeds = encode_classes(imagenet_classnames, templates=self.templates, embedder=self.text_embedder)
        closest_classes = find_closest_imagenet_classes(cifar_embeds, imagenet_embeds, imagenet_classnames, topk=1, return_indices=return_indices)
        return [closest[0][0] for closest in closest_classes]
    
    def _make_cond_label(self, labels, device, c_dim):
        if c_dim == 0:
            return torch.zeros([self.batch_size, 0], device=device)
        if self.imagenet:
            labels = self._compute_imagenet_class_index(labels, self.imagenet_classnames, return_indices=True)
            log.info(f"Mapped CIFAR labels to ImageNet classes {labels} for conditional generation.")
        label = one_hot_from_int(labels, batch_size=self.batch_size, num_classes=self.num_classes)
        return torch.tensor(label, dtype=torch.float, device=device)

    def _resize_biggan_candidate(self, candidate):
        # log.info(f"Candidate generated by BigGAN has shape {candidate.shape} and value range [{candidate.min().item():.4f}, {candidate.max().item():.4f}]")
        if self.cfg.generator.type != "biggan":
            return candidate
        if candidate.ndim != 4 or len(self.data_shape) < 3:
            return candidate
        target_h, target_w = self.data_shape[-2], self.data_shape[-1]
        if candidate.shape[-2:] == (target_h, target_w):
            return candidate
        candidate = F.interpolate(candidate, size=(target_h, target_w), mode="bilinear", align_corners=False)
        # log.info(f"Candidate generated by BigGAN has shape {candidate.shape} and value range [{candidate.min().item():.4f}, {candidate.max().item():.4f}]")
        return candidate

    def _resolve_biggan_weights(self):
        if self._biggan_weights_path is not None:
            return self._biggan_weights_path
        if not hasattr(self.cfg, "generator"):
            return None
        use_pretrained = bool(
            getattr(self.cfg.generator, "pretrained", False) or getattr(self.cfg.generator, "network_wts", None)
        )
        if not use_pretrained:
            return None
        model_name = biggan_model_name_for_output_dim(self.bigganconfig.output_dim)
        if model_name is None:
            log.info("No pretrained BigGAN available for output_dim=%s.", self.bigganconfig.output_dim)
            return None
        if self.bigganconfig.num_classes != 1000:
            log.info(
                "Skipping pretrained BigGAN weights for num_classes=%s (expected 1000).",
                self.bigganconfig.num_classes,
            )
            return None
        requested_path = getattr(self.cfg.generator, "network_wts", None)
        weights_path = resolve_pretrained_biggan_weights(model_name, requested_path=requested_path)
        if not weights_path:
            return None
        self.cfg.generator.network_wts = weights_path
        self._biggan_weights_path = weights_path
        return weights_path

    def _generate_candidate(self, netG, noise, labels):
        if self.cfg.generator.type == "biggan":
            # label = self._make_cond_label(labels, self.setup["device"], self.num_classes)
            # log.info(f"noise finite: {torch.isfinite(noise).all()} range [{noise.min().item():.4f}, {noise.max().item():.4f}]")
            # log.info(f"label finite: {torch.isfinite(label).all()} range [{label.min().item():.4f}, {label.max().item():.4f}]")
            candidate = netG(noise, labels, truncation=self.truncation)
            candidate = self._resize_biggan_candidate(candidate)
            # log.info(f"Candidate generated by BigGAN has shape {candidate.shape} and value range [{candidate.min().item():.4f}, {candidate.max().item():.4f}]")
            return candidate
        if self.cfg.generator.type in ("styleganxl", "stylegan_xl"):
            # label = self._make_cond_label(labels, self.setup["device"], netG.c_dim)
            return netG(noise, labels, truncation_psi=self.truncation, noise_mode='random')
        if self.cfg.generator.type in ("stylegan2", "stylegan2_ada"):
            # label = self._make_cond_label(labels, self.setup["device"], netG.c_dim)
            return netG(noise, labels, truncation_psi=self.truncation, noise_mode='random')
        raise ValueError(f"Unknown generator type {self.cfg.generator.type}.")

    def reconstruct(self, server_payload, shared_data, server_secrets=None, initial_data=None, dryrun=False):
        # Initialize stats module for later usage:
        rec_models, labels, stats = self.prepare_attack(server_payload, shared_data)
        num_classes = server_payload[0]["metadata"]["classes"]
        log.info(f"Number of classes in the client dataset is {num_classes}.")
        
        # Main reconstruction loop starts here:
        scores = torch.zeros(self.cfg.restarts.num_trials)
        candidate_solutions = []
        try:
            if not self.parallel_trials:
                for trial in range(self.cfg.restarts.num_trials):
                    candidate_solutions += [
                        self._run_trial(rec_models, shared_data, labels, stats, trial, initial_data, dryrun)
                    ]
                    scores[trial] = self._score_trial(candidate_solutions[trial], labels, rec_models, shared_data)
            else:
                candidate_solutions = self._run_parallel_trials(
                    rec_models, shared_data, labels, stats, initial_data, dryrun, self.cfg.restarts.num_trials
                )
                for trial in range(self.cfg.restarts.num_trials):
                    scores[trial] = self._score_trial(candidate_solutions[trial], labels, rec_models, shared_data)
        except KeyboardInterrupt:
            print("Trial procedure manually interruped.")
            pass
        optimal_solution = self._select_optimal_reconstruction(candidate_solutions, scores, stats)
        reconstructed_data = dict(data=optimal_solution, labels=labels)
        if server_payload[0]["metadata"].modality == "text":
            reconstructed_data = self._postprocess_text_data(reconstructed_data)
        if "ClassAttack" in server_secrets:
            # Only a subset of images was actually reconstructed:
            true_num_data = server_secrets["ClassAttack"]["true_num_data"]
            reconstructed_data["data"] = torch.zeros([true_num_data, *self.data_shape], **self.setup)
            reconstructed_data["data"][server_secrets["ClassAttack"]["target_indx"]] = optimal_solution
            reconstructed_data["labels"] = server_secrets["ClassAttack"]["all_labels"]
        return reconstructed_data, stats

    def _run_trial(self, rec_model, shared_data, labels, stats, trial, initial_data=None, dryrun=False):
        """Run a single reconstruction trial."""
        # Initialize losses:
        for regularizer in self.regularizers:
            regularizer.initialize(rec_model, shared_data, labels)
        self.objective.initialize(self.loss_fn, self.cfg.impl, shared_data[0]["metadata"]["local_hyperparams"])
        # Initialize candidate reconstruction data
        
        self.batch_size = shared_data[0]["metadata"]["num_data_points"]
        # GISMN Does not require a pre-trained BigGAN. It uses pre-trained StyleGAN-XL models instead, 
        # but we are primarily using BigGAN for now. Will change to StyleGAN-XL later
        # so we can directly use the architecture and randomly initialized weights for the attack
        # self.netG = BigGAN(self.bigganconfig).to(self.setup["device"])
        if self.cfg.generator.type == "biggan":
            self.netG = BigGAN(self.bigganconfig).to(self.setup["device"])
            weights_path = self._resolve_biggan_weights()
            log.info(f"BigGAN weights path resolved to {weights_path}")
            if weights_path:
                if self._biggan_state_dict is None:
                    self._biggan_state_dict = torch.load(weights_path, map_location="cpu")
                self.netG.load_state_dict(self._biggan_state_dict, strict=False)
                log.info(f"Loaded BigGAN weights from {weights_path}")
            else:
                log.info(f"Randomly Initializing BigGAN weights since no pretrained weights found for output_dim={self.bigganconfig.output_dim} and num_classes={self.bigganconfig.num_classes}.")
                self.netG.init_weights_normal(mean=0.0, std=1.0)  # Initialize the generator weights with normal distribution
            # Define noise
            self.noise = truncated_noise_sample(batch_size=self.batch_size, 
                                            dim_z=self.netG.config.z_dim, 
                                            truncation=self.truncation)
        elif self.cfg.generator.type == "styleganxl":
            log.info(f"Loading StyleGAN-XL generator with weights from {self.cfg.generator.network_wts}")
            self.netG = build_stylegan_xl_generator(self.setup["device"], network_pkl=self.cfg.generator.network_wts)
            # Define noise
            self.noise = truncated_noise_sample(batch_size=self.batch_size, 
                                            dim_z=self.netG.z_dim, 
                                            truncation=self.truncation)
        elif self.cfg.generator.type in ("stylegan2", "stylegan2_ada"):
            log.info(f"Loading StyleGAN2-ADA generator with weights from {self.cfg.generator.network_wts}")
            self.netG = build_stylegan2_ada_generator(self.setup["device"], network_pkl=self.cfg.generator.network_wts)
            # Define noise
            self.noise = truncated_noise_sample(batch_size=self.batch_size,
                                            dim_z=self.netG.z_dim,
                                            truncation=self.truncation)
        #.from_pretrained('biggan-deep-128', cache_dir='./pretrained/models_128').to(self.setup["device"])
        self.netG.eval()
        # self.netG = BigGAN(self.bigganconfig).to(self.setup["device"])
        log.info("The number of parameters in the generator is {}".format(sum(p.numel() for p in self.netG.parameters())))
        # log.info(f"self.netG.embeddings.weight.shape = {self.netG.embeddings.weight.shape}")
        for p in self.netG.parameters():
            p.requires_grad = False
        # Define noise        
        try:
            # For BigGAN model
            self.noise = truncated_noise_sample(batch_size=self.batch_size, 
                                                dim_z=self.netG.config.z_dim, 
                                                truncation=self.truncation)
        except AttributeError:
            # For StyleGAN-XL model
            self.noise = truncated_noise_sample(batch_size=self.batch_size, 
                                                dim_z=self.netG.z_dim, 
                                                truncation=self.truncation)
        self.noise = torch.tensor(self.noise, **self.setup).detach().clone().requires_grad_(True)

        minimal_value_so_far = torch.as_tensor(float("inf"), **self.setup)

        log.info(f"c_dim for the generator is {self.netG.c_dim}")
        if self.cfg.generator.type == "biggan":
            labels = self._make_cond_label(labels, self.setup["device"], self.num_classes)
        elif self.cfg.generator.type == "styleganxl":
            labels = self._make_cond_label(labels, self.setup["device"], self.netG.c_dim)
        elif self.cfg.generator.type in ("stylegan2", "stylegan2_ada"):
            labels = self._make_cond_label(labels, self.setup["device"], self.netG.c_dim)

        with torch.no_grad():
            best_candidate = self._generate_candidate(self.netG, self.noise, labels).detach().clone()

        # Initialize optimizers
        # In CI-Net, we optimzie the parameters ofthe Generator network instead of the image pixels directly
        optimizer, scheduler = self._init_optimizer([self.noise])
        current_wallclock = time.time()
        try:
            for iteration in range(self.cfg.optim.max_iterations):
                self.curr_iteration = iteration
                closure = self._compute_objective(self.netG,
                                                  self.noise, 
                                                    labels, 
                                                    rec_model, 
                                                    optimizer, 
                                                    shared_data, # Contains both gradients and posembed gradients
                                                    iteration)
                objective_value, task_loss = optimizer.step(closure), self.current_task_loss
                scheduler.step()

                with torch.no_grad():
                    # Project into image space
                    if self.cfg.generator.type == "biggan":
                        # label = self._make_cond_label(labels, self.setup["device"], self.num_classes)
                        candidate = self.netG(self.noise, labels, truncation=self.truncation)
                        candidate = self._resize_biggan_candidate(candidate)
                        # log.info(f"Candidate generated by BigGAN has shape {candidate.shape} and value range [{candidate.min().item():.4f}, {candidate.max().item():.4f}]")
                    elif self.cfg.generator.type == "styleganxl":
                        # label = self._make_cond_label(labels, self.setup["device"], self.netG.c_dim)
                        candidate = self.netG(self.noise, labels, truncation_psi=self.truncation, noise_mode='random')
                    elif self.cfg.generator.type in ("stylegan2", "stylegan2_ada"):
                        # label = self._make_cond_label(labels, self.setup["device"], self.netG.c_dim)
                        candidate = self.netG(self.noise, labels, truncation_psi=self.truncation, noise_mode='random')
                    if self.cfg.optim.boxed:
                        candidate.data = torch.max(torch.min(candidate, (1 - self.dm) / self.ds), -self.dm / self.ds)
                    if objective_value < minimal_value_so_far:
                        minimal_value_so_far = objective_value.detach()
                        best_candidate = candidate.detach().clone()

                if iteration + 1 == self.cfg.optim.max_iterations or iteration % self.cfg.optim.callback == 0:
                    timestamp = time.time()
                    log.info(
                        f"| It: {iteration + 1} | Rec. loss: {objective_value.item():2.4f} | "
                        f" Task loss: {task_loss.item():2.4f} | T: {timestamp - current_wallclock:4.2f}s"
                    )
                    current_wallclock = timestamp
                    # log.info("Norm of Generator parameters after {} iters of optimization: {}".format(iteration, sum(p.norm().item() for p in self.netG.parameters())))

                if not torch.isfinite(objective_value):
                    log.info(f"Recovery loss is non-finite in iteration {iteration}. Cancelling reconstruction!")
                    break

                stats[f"Trial_{trial}_Val"].append(objective_value.item())

                if dryrun:
                    break
        except KeyboardInterrupt:
            print(f"Recovery interrupted manually in iteration {iteration}!")
            pass

        self.netG = None  # Free up memory

        return best_candidate.detach()

    def _run_parallel_trials(self, rec_model, shared_data, labels, stats, initial_data, dryrun, num_trials):
        """Run multiple GISMN trials in parallel for parallel regularizers."""
        for regularizer in self.regularizers:
            regularizer.initialize(rec_model, shared_data, labels)
        self.objective.initialize(self.loss_fn, self.cfg.impl, shared_data[0]["metadata"]["local_hyperparams"])

        self.batch_size = shared_data[0]["metadata"]["num_data_points"]

        netGs, noises = [], []
        for _ in range(num_trials):
            if self.cfg.generator.type == "biggan":
                netG = BigGAN(self.bigganconfig).to(self.setup["device"])
                weights_path = self._resolve_biggan_weights()
                if weights_path:
                    if self._biggan_state_dict is None:
                        self._biggan_state_dict = torch.load(weights_path, map_location="cpu")
                        netG.load_state_dict(self._biggan_state_dict, strict=False)
                    log.info(f"Loaded BigGAN weights from {weights_path}")
                else:
                    log.info(f"Randomly Initializing BigGAN weights since no pretrained weights found for output_dim={self.bigganconfig.output_dim} and num_classes={self.bigganconfig.num_classes}.")
                    netG.init_weights_normal(mean=0.0, std=1.0)  # Initialize the generator weights with normal distribution
                noise = truncated_noise_sample(
                    batch_size=self.batch_size, dim_z=netG.config.z_dim, truncation=self.truncation
                )
                noise = torch.tensor(noise, **self.setup)
            elif self.cfg.generator.type in ("styleganxl", "stylegan_xl"):
                netG = build_stylegan_xl_generator(self.setup["device"], network_pkl=self.cfg.generator.network_wts)
                noise = truncated_noise_sample(batch_size=self.batch_size, dim_z=netG.z_dim, truncation=self.truncation)
                noise = torch.tensor(noise, **self.setup)
            elif self.cfg.generator.type in ("stylegan2", "stylegan2_ada"):
                netG = build_stylegan2_ada_generator(self.setup["device"], network_pkl=self.cfg.generator.network_wts)
                noise = truncated_noise_sample(batch_size=self.batch_size, dim_z=netG.z_dim, truncation=self.truncation)
                noise = torch.tensor(noise, **self.setup)
            else:
                raise ValueError(f"Unknown generator type {self.cfg.generator.type}.")
            netG.eval()
            for p in netG.parameters():
                p.requires_grad = False
            netGs.append(netG)
            noises.append(noise.detach().clone().requires_grad_(True))
        
        if self.cfg.generator.type == "biggan":
            labels = self._make_cond_label(labels, self.setup["device"], self.num_classes)
        elif self.cfg.generator.type == "styleganxl":
            labels = self._make_cond_label(labels, self.setup["device"], netGs[-1].c_dim)
        elif self.cfg.generator.type in ("stylegan2", "stylegan2_ada"):
            labels = self._make_cond_label(labels, self.setup["device"], netGs[-1].c_dim)

        with torch.no_grad():
            best_candidates = [self._generate_candidate(netG, noise, labels).detach().clone() for netG, noise in zip(netGs, noises)]
        minimal_values_so_far = torch.as_tensor([float("inf")] * num_trials, **self.setup)

        optimizers, schedulers = [], []
        for noise in noises:
            optimizer, scheduler = self._init_optimizer([noise])
            optimizers.append(optimizer)
            schedulers.append(scheduler)

        parallel_regularizers = getattr(self, "parallel_regularizers", [])
        current_wallclock = time.time()
        try:
            for iteration in range(self.cfg.optim.max_iterations):
                self.curr_iteration = iteration
                for optimizer in optimizers:
                    optimizer.zero_grad()

                objective_values, task_losses, candidates = [], [], []
                for netG, noise in zip(netGs, noises):
                    candidate = self._generate_candidate(netG, noise, labels)
                    if self.cfg.differentiable_augmentations:
                        candidate_augmented = self.augmentations(candidate)
                    else:
                        candidate_augmented = candidate
                        candidate_augmented.data = self.augmentations(candidate.data)

                    total_objective = 0
                    total_task_loss = 0
                    for model, data in zip(rec_model, shared_data):
                        objective, task_loss = self.objective(model, data["gradients"], candidate_augmented, labels)
                        total_objective += objective
                        total_task_loss += task_loss
                    if self.curr_iteration >= self.cfg.optim.max_iterations * 4 // 9:
                        for regularizer in self.regularizers:
                            total_objective += regularizer(candidate_augmented)

                    objective_values.append(total_objective)
                    task_losses.append(total_task_loss)
                    candidates.append(candidate)

                for regularizer in parallel_regularizers:
                    reg_loss = regularizer(candidates)
                    for idx, reg in enumerate(reg_loss):
                        objective_values[idx] = objective_values[idx] + reg

                total_objective = sum(objective_values)
                if total_objective.requires_grad:
                    candidate_grads = torch.autograd.grad(total_objective, candidates, create_graph=False)
                    grad_list = list(candidate_grads)
                    with torch.no_grad():
                        for idx, candidate_grad in enumerate(grad_list):
                            if self.cfg.optim.langevin_noise > 0:
                                step_size = optimizers[idx].param_groups[0]["lr"]
                                noise_map = torch.randn_like(candidate_grad)
                                candidate_grad += self.cfg.optim.langevin_noise * step_size * noise_map
                            if self.cfg.optim.grad_clip is not None:
                                grad_norm = candidate_grad.norm()
                                if grad_norm > self.cfg.optim.grad_clip:
                                    candidate_grad.mul_(self.cfg.optim.grad_clip / (grad_norm + 1e-6))
                            if self.cfg.optim.signed is not None:
                                if self.cfg.optim.signed == "soft":
                                    scaling_factor = 1 - iteration / self.cfg.optim.max_iterations
                                    candidate_grad.mul_(scaling_factor).tanh_().div_(scaling_factor)
                                elif self.cfg.optim.signed == "hard":
                                    candidate_grad.sign_()
                    torch.autograd.backward(candidates, grad_list)

                for optimizer, scheduler in zip(optimizers, schedulers):
                    optimizer.step()
                    scheduler.step()

                with torch.no_grad():
                    for idx, candidate in enumerate(candidates):
                        if self.cfg.optim.boxed:
                            candidate.data = torch.max(torch.min(candidate, (1 - self.dm) / self.ds), -self.dm / self.ds)
                        if objective_values[idx] < minimal_values_so_far[idx]:
                            minimal_values_so_far[idx] = objective_values[idx].detach()
                            best_candidates[idx] = candidate.detach().clone()

                if iteration + 1 == self.cfg.optim.max_iterations or iteration % self.cfg.optim.callback == 0:
                    timestamp = time.time()
                    for idx, (objective_value, task_loss) in enumerate(zip(objective_values, task_losses)):
                        log.info(
                            f"| Trial: {idx} | It: {iteration + 1} | Rec. loss: {objective_value.item():2.4f} | "
                            f" Task loss: {task_loss.item():2.4f} | T: {timestamp - current_wallclock:4.2f}s"
                        )
                    current_wallclock = timestamp

                if not all(torch.isfinite(obj) for obj in objective_values):
                    log.info(f"Recovery loss is non-finite in iteration {iteration}. Cancelling reconstruction!")
                    break
                for trial in range(num_trials):
                    stats[f"Trial_{trial}_Val"].append(objective_values[trial].item())
                if dryrun:
                    break
        except KeyboardInterrupt:
            print(f"Recovery interrupted manually in iteration {iteration}!")
            pass

        self.netG = None
        return [candidate.detach() for candidate in best_candidates]

    def _compute_objective(self, netG, noise, labels, rec_model, optimizer, shared_data, iteration):
        def closure():
            optimizer.zero_grad()
            if self.cfg.generator.type == "biggan":
                # label = self._make_cond_label(labels, self.setup["device"], self.num_classes)
                candidate = netG(noise, labels, truncation=self.truncation)
                candidate = self._resize_biggan_candidate(candidate)
            elif self.cfg.generator.type == "styleganxl":
                # label = self._make_cond_label(labels, self.setup["device"], netG.c_dim)
                candidate = netG(noise, labels, truncation_psi=self.truncation, noise_mode='random')
            elif self.cfg.generator.type in ("stylegan2", "stylegan2_ada"):
                # label = self._make_cond_label(labels, self.setup["device"], netG.c_dim)
                candidate = netG(noise, labels, truncation_psi=self.truncation, noise_mode='random')

            if self.cfg.differentiable_augmentations:
                candidate_augmented = self.augmentations(candidate)
            else:
                candidate_augmented = candidate
                candidate_augmented.data = self.augmentations(candidate.data)

            total_objective = 0
            total_task_loss = 0
            for model, data in zip(rec_model, shared_data):

                objective, task_loss = self.objective(model, data["gradients"], candidate_augmented, labels)
                total_objective += objective
                total_task_loss += task_loss
            
            if self.curr_iteration >= self.cfg.optim.max_iterations * 4 // 9:
                # Add regularization only in the later stages of optimization, to allow more freedom in the initial stages
                for regularizer in self.regularizers:
                    total_objective += regularizer(candidate_augmented)

            if total_objective.requires_grad:
                candidate_grad = torch.autograd.grad(total_objective, noise, create_graph=False)[0]
                with torch.no_grad():
                    if self.cfg.optim.langevin_noise > 0:
                        step_size = optimizer.param_groups[0]["lr"]
                        noise_map = torch.randn_like(candidate_grad)
                        candidate_grad += self.cfg.optim.langevin_noise * step_size * noise_map
                    if self.cfg.optim.grad_clip is not None:
                        grad_norm = candidate_grad.norm()
                        if grad_norm > self.cfg.optim.grad_clip:
                            candidate_grad.mul_(self.cfg.optim.grad_clip / (grad_norm + 1e-6))
                    if self.cfg.optim.signed is not None:
                        if self.cfg.optim.signed == "soft":
                            scaling_factor = (
                                1 - iteration / self.cfg.optim.max_iterations
                            )  # just a simple linear rule for now
                            candidate_grad.mul_(scaling_factor).tanh_().div_(scaling_factor)
                        elif self.cfg.optim.signed == "hard":
                            candidate_grad.sign_()
                        else:
                            pass
                noise.backward(candidate_grad)
                # candidate_grad = torch.autograd.grad(total_objective, netG.parameters(), create_graph=False)
                # for p, g in zip(netG.parameters(), candidate_grad):
                #     p.grad = g

            self.current_task_loss = total_task_loss  # Side-effect this because of L-BFGS closure limitations :<
            return total_objective

        return closure
