""" Implements the CGIR attacks on ViT models.
This method uses L2 loss optimization for the parameter gradients,
And the positional embedding gradients are also recovered using cosine similarity loss
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time

from .optimization_based_attack import OptimizationBasedAttacker
from .auxiliaries.objectives import Euclidean, CosineSimilarity, objective_lookup
from .auxiliaries.augmentations import augmentation_lookup

from .auxiliaries.BigGAN.model import BigGAN, Generator
from .auxiliaries.BigGAN.config import BigGANConfig, BigGAN32, BigGAN128, BigGAN256, BigGAN512
from .auxiliaries.BigGAN.utils import truncated_noise_sample, save_as_images, one_hot_from_names, one_hot_from_int

from .auxiliaries.stylegan_xl import legacy, dnnlib
from .auxiliaries.stylegan_xl.torch_utils import misc

from .auxiliaries.dcgan.dcgan import Generator as DCGANGenerator

import logging

log = logging.getLogger(__name__)

def _unwrap_state_dict(state):
    if isinstance(state, dict):
        for key in ['state_dict', 'model', 'netG', 'generator', 'G_ema', 'G', 'g_ema']:
            if key in state and isinstance(state[key], dict):
                return state[key]
    return state

def _strip_module_prefix(state_dict):
    if not isinstance(state_dict, dict):
        return state_dict
    if not any(k.startswith('module.') for k in state_dict.keys()):
        return state_dict
    return {k.replace('module.', '', 1): v for k, v in state_dict.items()}

def _load_state_dict_forgiving(model, state_dict):
    state_dict = _strip_module_prefix(state_dict)
    model_sd = model.state_dict()
    filtered = {}
    skipped = []
    for k, v in state_dict.items():
        if k in model_sd and model_sd[k].shape == v.shape:
            filtered[k] = v
        else:
            skipped.append((k, getattr(v, 'shape', None), model_sd.get(k).shape if k in model_sd else None))
    missing, unexpected = model.load_state_dict(filtered, strict=False)
    if skipped:
        log.warning("Skipped %d generator weights due to shape/key mismatch.", len(skipped))
    if missing:
        log.warning("Generator missing %d keys after load.", len(missing))
    if unexpected:
        log.warning("Generator got %d unexpected keys after load.", len(unexpected))

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

class GenImagePriorAttacker(OptimizationBasedAttacker):
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

    def _resize_dcgan_output(self, x):
        if self.cfg.generator.type != "dcgan":
            return x
        if x.ndim != 4 or len(self.data_shape) < 3:
            return x
        target_h, target_w = self.data_shape[-2], self.data_shape[-1]
        if x.shape[-2:] != (target_h, target_w):
            return F.interpolate(x, size=(target_h, target_w), mode="bilinear", align_corners=False)
        return x

    def reconstruct(self, server_payload, shared_data, server_secrets=None, initial_data=None, dryrun=False):
        # Initialize stats module for later usage:
        rec_models, labels, stats = self.prepare_attack(server_payload, shared_data)
        self.num_classes = server_payload[0]["metadata"]["classes"]
        log.info(f"Number of classes in the dataset is {self.num_classes}.")
        if self.cfg.generator.type == "biggan":
            if hasattr(server_payload[0]["metadata"], "name") and "CIFAR" in server_payload[0]["metadata"].name:
                self.bigganconfig = BigGAN32
            elif hasattr(server_payload[0]["metadata"], "name") and "ImageNet" in server_payload[0]["metadata"].name:
                self.bigganconfig = BigGAN256
            self.bigganconfig.num_classes = self.num_classes
        
        # Main reconstruction loop starts here:
        scores = torch.zeros(self.cfg.restarts.num_trials)
        candidate_solutions = []
        try:
            for trial in range(self.cfg.restarts.num_trials):
                candidate_solutions += [
                    self._run_trial(rec_models, shared_data, labels, stats, trial, initial_data, dryrun)
                ]
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
        # GIRG Does not require a pre-trained BigGAN
        # so we can directly use the architecture and randomly initialized weights for the attack
        if self.cfg.generator.type == "biggan":
            self.netG = BigGAN(self.bigganconfig).to(self.setup["device"])
            # Define noise
            self.noise = truncated_noise_sample(batch_size=self.batch_size, 
                                            dim_z=self.netG.config.z_dim, 
                                            truncation=self.truncation)
        elif self.cfg.generator.type == "stylegan_xl":
            self.netG = build_stylegan_xl_generator(self.setup["device"], network_pkl=self.cfg.generator.network_wts)
            # Define noise
            self.noise = truncated_noise_sample(batch_size=self.batch_size, 
                                            dim_z=self.netG.z_dim, 
                                            truncation=self.truncation)
        elif self.cfg.generator.type == "dcgan":
            self.netG = DCGANGenerator(ngpu=1, nz=self.cfg.generator.z_dim, ngf=self.cfg.generator.ngf, nc=self.cfg.generator.nc).to(self.setup["device"])
            state = torch.load(self.cfg.generator.network_wts, map_location=self.setup["device"])
            # state = _unwrap_state_dict(state)
            # _load_state_dict_forgiving(self.netG, state)
            self.netG.load_state_dict(state)
            # Define noise
            # self.noise = truncated_noise_sample(batch_size=self.batch_size, 
            #                                 dim_z=self.cfg.generator.z_dim, 
            #                                 truncation=self.truncation)
            self.noise = torch.randn(self.batch_size, self.cfg.generator.z_dim, 1, 1, device=self.setup["device"])

        log.info("The number of parameters in the generator is {}".format(sum(p.numel() for p in self.netG.parameters())))
        log.info("Shape of Noise is {}".format(self.noise.shape))
        # log.info(f"self.netG.embeddings.weight.shape = {self.netG.embeddings.weight.shape}")
        for p in self.netG.parameters():
            p.requires_grad = False

        if isinstance(self.noise, np.ndarray):
            self.noise = torch.tensor(self.noise, **self.setup).detach().clone().requires_grad_(True)
        else:
            self.noise = self.noise.detach().clone().requires_grad_(True)
        
        minimal_value_so_far = torch.as_tensor(float("inf"), **self.setup)
        # Initialize optimizers
        # In GIAS, we optimzie the latent code first
        optimizer, scheduler = self._init_optimizer([self.noise])
        current_wallclock = time.time()
        try:
            for iteration in range(self.cfg.optim.max_iterations):
                closure = self._compute_objective_latent(self.netG, 
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
                    with torch.no_grad():
                        label = one_hot_from_int(labels, batch_size=self.batch_size, num_classes=self.num_classes)
                        label = torch.tensor(label, dtype=torch.float).to(self.setup["device"])
                        if self.cfg.generator.type == "biggan":
                            # For BigGAN model
                            candidate = self.netG(self.noise, label, truncation=self.truncation)
                        elif self.cfg.generator.type == "stylegan_xl":
                            # For StyleGAN-XL model
                            candidate = self.netG(self.noise, label, truncation_psi=self.truncation, noise_mode='random')
                        elif self.cfg.generator.type == "dcgan":
                            candidate = self.netG(self.noise)
                        candidate = self._resize_dcgan_output(candidate)
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
            
            ###################################################
            ### parameter OPTIMIZATION STARTS HERE  ##########
            ###################################################
            log.info("Starting fine level optimization for trial {} with the best candidate from coarse optimization.".format(trial))
            for p in self.netG.parameters():
                p.requires_grad = True

            self.noise = self.noise.requires_grad_(False)

            with torch.no_grad():
                label = one_hot_from_int(labels, batch_size=self.batch_size, num_classes=self.num_classes)
                label = torch.tensor(label, dtype=torch.float).to(self.setup["device"])
                if self.cfg.generator.type == "biggan":
                    # For BigGAN model
                    candidate = self.netG(self.noise, label, truncation=self.truncation)
                elif self.cfg.generator.type == "stylegan_xl":
                    # For StyleGAN-XL model
                    candidate = self.netG(self.noise, label, truncation_psi=self.truncation, noise_mode='random')
                elif self.cfg.generator.type == "dcgan":
                    candidate = self.netG(self.noise)
            candidate = self._resize_dcgan_output(candidate)
            
            best_candidate = candidate.detach().clone()
            minimal_value_so_far = torch.as_tensor(float("inf"), **self.setup)

            # Initialize optimizers
            # Since _init_optimizer only looks into self.cfg.optim,
            # we need to manually overwrite the optimization parameters here to use the ones for parameter optimization
            self.cfg.optim.optimizer = self.cfg.optim_w.optimizer
            self.cfg.optim.step_size = self.cfg.optim_w.step_size
            self.cfg.optim.max_iterations = self.cfg.optim_w.max_iterations
            self.cfg.optim.callback = self.cfg.optim_w.callback

            optimizer, scheduler = self._init_optimizer([p for p in self.netG.parameters() if p.requires_grad])

            current_wallclock = time.time()
            for iteration in range(self.cfg.optim_w.max_iterations):
                closure = self._compute_objective(self.netG, 
                                                    self.noise,
                                                    labels, rec_model, optimizer, shared_data, iteration)
                objective_value, task_loss = optimizer.step(closure), self.current_task_loss
                scheduler.step()

                with torch.no_grad():
                    # Project into image space
                    label = one_hot_from_int(labels, batch_size=self.batch_size, num_classes=self.num_classes)
                    label = torch.tensor(label, dtype=torch.float).to(self.setup["device"])
                    if self.cfg.generator.type == "biggan":
                        # For BigGAN model
                        candidate = self.netG(self.noise, label, truncation=self.truncation)
                    elif self.cfg.generator.type == "stylegan_xl":
                        # For StyleGAN-XL model
                        candidate = self.netG(self.noise, label, truncation_psi=self.truncation, noise_mode='random')
                    elif self.cfg.generator.type == "dcgan":
                        candidate = self.netG(self.noise)
                    candidate = self._resize_dcgan_output(candidate)

                    if self.cfg.optim.boxed:
                        candidate.data = torch.max(torch.min(candidate, (1 - self.dm) / self.ds), -self.dm / self.ds)
                    if objective_value < minimal_value_so_far:
                        minimal_value_so_far = objective_value.detach()
                        best_candidate = candidate.detach().clone()

                if iteration + 1 == self.cfg.optim.fine_iterations or iteration % self.cfg.optim.callback == 0:
                    timestamp = time.time()
                    log.info(
                        f"| It: {iteration + 1} | Rec. loss: {objective_value.item():2.4f} | "
                        f" Task loss: {task_loss.item():2.4f} | T: {timestamp - current_wallclock:4.2f}s"
                    )
                    current_wallclock = timestamp

                if not torch.isfinite(objective_value):
                    log.info(f"Recovery loss is non-finite in iteration {iteration}. Cancelling reconstruction!")
                    break

                stats[f"Trial_{trial}_Val"].append(objective_value.item())

                if dryrun:
                    break

        except KeyboardInterrupt:
            print(f"Recovery interrupted manually in iteration {iteration}!")
            pass

        self.netG = None # Free up memory

        return best_candidate.detach()

    def _compute_objective(self, netG, noise, labels, rec_model, optimizer, shared_data, iteration):
        def closure():
            optimizer.zero_grad()
            # GENERATION
            label = one_hot_from_int(labels, batch_size=self.batch_size, num_classes=self.num_classes)
            # noise_t = torch.tensor(noise, dtype=torch.float).to(self.setup["device"])
            label = torch.tensor(label, dtype=torch.float).to(self.setup["device"])
            # cond_vector = torch.cat((noise_t, netG.embeddings(label)), dim=1)
            # candidate = netG.generator(cond_vector, self.truncation)
            with torch.no_grad():
                if self.cfg.generator.type == "biggan":
                    # For BigGAN model
                    candidate = netG(noise, label, truncation=self.truncation)
                elif self.cfg.generator.type == "stylegan_xl":
                    # For StyleGAN-XL model
                    candidate = netG(noise, label, truncation_psi=self.truncation, noise_mode='random')
                elif self.cfg.generator.type == "dcgan":
                    candidate = netG(noise)
            candidate = self._resize_dcgan_output(candidate)
            # log.info(f"candidate.shape = {candidate.shape}")

            if self.cfg.differentiable_augmentations:
                candidate_augmented = self.augmentations(candidate)
            else:
                candidate_augmented = candidate
                candidate_augmented.data = self.augmentations(candidate.data)

            total_objective = 0
            total_task_loss = 0
            for model, data in zip(rec_model, shared_data):
                # data["gradients"] is a list of gradients for each layer
                objective, task_loss = self.objective(model, data["gradients"], candidate_augmented, labels)
                total_objective += objective
                total_task_loss += task_loss
            for regularizer in self.regularizers:
                    total_objective += regularizer(candidate_augmented)

            if total_objective.requires_grad:
                # candidate_grad = torch.autograd.grad(total_objective, candidate, create_graph=False)[0]
                # with torch.no_grad():
                #     if self.cfg.optim.langevin_noise > 0:
                #         step_size = optimizer.param_groups[0]["lr"]
                #         noise_map = torch.randn_like(candidate_grad)
                #         candidate_grad += self.cfg.optim.langevin_noise * step_size * noise_map
                #     if self.cfg.optim.grad_clip is not None:
                #         grad_norm = candidate_grad.norm()
                #         if grad_norm > self.cfg.optim.grad_clip:
                #             candidate_grad.mul_(self.cfg.optim.grad_clip / (grad_norm + 1e-6))
                #     if self.cfg.optim.signed is not None:
                #         if self.cfg.optim.signed == "soft":
                #             scaling_factor = (
                #                 1 - iteration / self.cfg.optim.max_iterations
                #             )  # just a simple linear rule for now
                #             candidate_grad.mul_(scaling_factor).tanh_().div_(scaling_factor)
                #         elif self.cfg.optim.signed == "hard":
                #             candidate_grad.sign_()
                #         else:
                #             pass
                # candidate.backward(candidate_grad)
                candidate_grad = torch.autograd.grad(total_objective, netG.parameters(), create_graph=False)
                for p, g in zip(netG.parameters(), candidate_grad):
                    p.grad = g

            self.current_task_loss = total_task_loss  # Side-effect this because of L-BFGS closure limitations :<
            return total_objective

        return closure
    
    def _compute_objective_latent(self, netG, noise, labels, rec_model, optimizer, shared_data, iteration):
        def closure():
            optimizer.zero_grad()
            label = one_hot_from_int(labels, batch_size=self.batch_size, num_classes=self.num_classes)
            label = torch.tensor(label, dtype=torch.float).to(self.setup["device"])
            if self.cfg.generator.type == "biggan":
                # For BigGAN model
                candidate = netG(noise, label, truncation=self.truncation)
            elif self.cfg.generator.type == "stylegan_xl":
                # For StyleGAN-XL model
                candidate = netG(noise, label, truncation_psi=self.truncation, noise_mode='random')
            elif self.cfg.generator.type == "dcgan":
                candidate = netG(noise)
            candidate = self._resize_dcgan_output(candidate)
            # log.info(f"candidate.shape = {candidate.shape}")

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

    
