""" Implements the GIRG attacks on ViT models.
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

from .auxiliaries.BigGAN.model import BigGAN, Generator
from .auxiliaries.BigGAN.config import BigGANConfig, BigGAN32, BigGAN128, BigGAN256, BigGAN512
from .auxiliaries.BigGAN.utils import truncated_noise_sample, save_as_images, one_hot_from_names, one_hot_from_int

import logging

log = logging.getLogger(__name__)

class ConditionalGANGradMatchingAttacker(OptimizationBasedAttacker):
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

    def reconstruct(self, server_payload, shared_data, server_secrets=None, initial_data=None, dryrun=False):
        # Initialize stats module for later usage:
        rec_models, labels, stats = self.prepare_attack(server_payload, shared_data)
        self.num_classes = server_payload[0]["metadata"]["classes"]
        log.info(f"Number of classes in the dataset is {self.num_classes}.")
        if hasattr(server_payload[0]["metadata"], "name") and "CIFAR" in server_payload[0]["metadata"].name:
            self.bigganconfig = BigGAN32
        elif hasattr(server_payload[0]["metadata"], "name") and "ImageNet" in server_payload[0]["metadata"].name:
            self.bigganconfig = BigGAN256
        self.bigganconfig.num_classes = self.num_classes
        # GIRG Does not require a pre-trained BigGAN
        # so we can directly use the architecture and randomly initialized weights for the attack
        self.netG = BigGAN(self.bigganconfig).to(self.setup["device"])
        log.info("The number of parameters in the generator is {}".format(sum(p.numel() for p in self.netG.parameters())))
        # log.info(f"self.netG.embeddings.weight.shape = {self.netG.embeddings.weight.shape}")
        for p in self.netG.parameters():
            p.requires_grad = True
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
        
        # Define noise        
        self.noise = truncated_noise_sample(batch_size=self.batch_size, 
                                            dim_z=self.netG.config.z_dim, 
                                            truncation=self.truncation)
        
        minimal_value_so_far = torch.as_tensor(float("inf"), **self.setup)
        # Initialize optimizers
        # In GIRG, we optimzie the parameters ofthe Generator network instead of the image pixels directly
        optimizer, scheduler = self._init_optimizer([p for p in self.netG.parameters()])
        current_wallclock = time.time()
        try:
            for iteration in range(self.cfg.optim.max_iterations):
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
                    candidate = self.netG(torch.tensor(self.noise, dtype=torch.float).to(self.setup["device"]), 
                                          torch.tensor(one_hot_from_int(labels, 
                                                                        batch_size=self.batch_size, 
                                                                        num_classes=self.num_classes), 
                                                        dtype=torch.float).to(self.setup["device"]), 
                                          self.truncation)
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

        return best_candidate.detach()

    def _compute_objective(self, netG, noise, labels, rec_model, optimizer, shared_data, iteration):
        def closure():
            optimizer.zero_grad()
            
            # GENERATION
            label = one_hot_from_int(labels, batch_size=self.batch_size, num_classes=self.num_classes)
            noise_t = torch.tensor(noise, dtype=torch.float).to(self.setup["device"])
            label = torch.tensor(label, dtype=torch.float).to(self.setup["device"])
            # cond_vector = torch.cat((noise_t, netG.embeddings(label)), dim=1)
            # candidate = netG.generator(cond_vector, self.truncation)
            candidate = netG(noise_t, label, self.truncation)
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
                candidate_grad = torch.autograd.grad(total_objective, candidate, create_graph=False)[0]
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
                candidate.backward(candidate_grad)
                # candidate_grad = torch.autograd.grad(total_objective, netG.parameters(), create_graph=False)
                # for p, g in zip(netG.parameters(), candidate_grad):
                #     p.grad = g

            self.current_task_loss = total_task_loss  # Side-effect this because of L-BFGS closure limitations :<
            return total_objective

        return closure
