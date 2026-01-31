"""Implementation for basic gradient inversion attacks.

This covers optimization-based reconstruction attacks as in Wang et al. "Beyond Infer-
ring Class Representatives: User-Level Privacy Leakage From Federated Learning."
and convers subsequent developments such as
* Zhu et al., "Deep Leakage from gradients",
* Geiping et al., "Inverting Gradients - How easy is it to break privacy in FL"
* ?
"""

import torch
import time

from .optimization_based_attack import OptimizationBasedAttacker
from .auxiliaries.regularizers import regularizer_lookup, TotalVariation
from .auxiliaries.objectives import Euclidean, CosineSimilarity, objective_lookup
from .auxiliaries.augmentations import augmentation_lookup

import logging

log = logging.getLogger(__name__)


class NonLinearSurrogateModelExtension(OptimizationBasedAttacker):
    """Implements a wide spectrum of optimization-based attacks."""

    def __init__(self, model, loss_fn, cfg_attack, setup=dict(dtype=torch.float, device=torch.device("cpu"))):
        super().__init__(model, loss_fn, cfg_attack, setup)
        self.t = torch.tensor(0.5, requires_grad=True, **setup)

    def _run_trial(self, rec_model, shared_data, labels, stats, trial, initial_data=None, dryrun=False):
        """Run a single reconstruction trial."""
        self.w0 = copy.deepcopy(rec_model).to(**setup).eval()
        self.wT = copy.deepcopy(rec_model).to(**setup).eval()
        for p0, pT, g in zip(self.w0.parameters(), self.w0.parameters(), shared_data[0]["gradients"]):
            pT.data = p0.data + g
        self.P1 = copy.deepcopy(self.w0).to(**setup)
        for p0, pT, p1 in zip(self.w0.parameters(), self.wT.parameters(), self.P1.parameters()):
            p1.data = 0.5*(p0.data + pT.data)
            p1.requires_grad = True
        self.P0 = copy.deepcopy(self.P1).to(**setup).eval()
        self.d = copy.deepcopy(self.w0).to(**setup)
        for p in self.d.parameters():
            p.data.fill_(1.0)
            p.requires_grad = True

        # Initialize losses:
        for regularizer in self.regularizers:
            regularizer.initialize(rec_model, shared_data, labels)
        self.objective.initialize(self.loss_fn, self.cfg.impl, shared_data[0]["metadata"]["local_hyperparams"])

        # Initialize candidate reconstruction data
        candidate = self._initialize_data([shared_data[0]["metadata"]["num_data_points"], *self.data_shape])
        if initial_data is not None:
            candidate.data = initial_data.data.clone().to(**self.setup)

        best_candidate = candidate.detach().clone()
        minimal_value_so_far = torch.as_tensor(float("inf"), **self.setup)

        # Initialize optimizers
        optimizer, scheduler = self._init_optimizer([candidate])
        t_optimizer = self._init_t_optimizer()
        d_optimizer = self._init_d_optimizer()
        p1_optimizer = self._init_p1_optimizer()
        current_wallclock = time.time()
        try:
            for iteration in range(self.cfg.optim.max_iterations):
                for p, p1, w0, wT in zip(rec_model.parameters(), self.P1.parameters(), self.w0.parameters(), self.wT.parameters()):
                    if self.cfg.optim.use_quad_bezier:
                        p.data = self._update_bezier_quadratic_(self.t, p1, w0, wT)
                    else:
                        p.data = self._update_bezier_linear(self.t, p1, w0, wT)
                closure = self._compute_objective(candidate, labels, rec_model, optimizer, shared_data, iteration)
                objective_value, task_loss = optimizer.step(closure), self.current_task_loss
                scheduler.step()
                t_optimizer.step()
                d_optimizer.step()
                p1_optimizer.step()

                with torch.no_grad():
                    # Project into image space
                    if self.cfg.optim.boxed:
                        candidate.data = torch.max(torch.min(candidate, (1 - self.dm) / self.ds), -self.dm / self.ds)
                        self.t.data = torch.clip(self.t.data, 0, 1)
                        self.d.data = torch.clip(self.d.data, 0.1, 10.0)
                        
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

    def _compute_objective(self, candidate, labels, rec_model, optimizer, shared_data, iteration):
        def closure():
            optimizer.zero_grad()

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
            # log.info(f"Objective Loss: {objective.item():2.4f}")
            # log.info(f"Number of regularizers: {len(self.regularizers)}")
            total_objective += self.regularizers[0](tensor=[p for p in self.P1.parameters()], 
                                                    target=[p for p in self.P0.parameters()])
            total_objective += self.regularizers[1](tensor=[p for p in self.d.parameters()], 
                                                    target=torch.tensor(1.0, **self.setup))

            if total_objective.requires_grad:
                total_objective.backward(inputs=candidate, create_graph=False)
            with torch.no_grad():
                if self.cfg.optim.langevin_noise > 0:
                    step_size = optimizer.param_groups[0]["lr"]
                    noise_map = torch.randn_like(candidate.grad)
                    candidate.grad += self.cfg.optim.langevin_noise * step_size * noise_map
                if self.cfg.optim.grad_clip is not None:
                    grad_norm = candidate.grad.norm()
                    if grad_norm > self.cfg.optim.grad_clip:
                        candidate.grad.mul_(self.cfg.optim.grad_clip / (grad_norm + 1e-6))
                if self.cfg.optim.signed is not None:
                    if self.cfg.optim.signed == "soft":
                        scaling_factor = (
                            1 - iteration / self.cfg.optim.max_iterations
                        )  # just a simple linear rule for now
                        candidate.grad.mul_(scaling_factor).tanh_().div_(scaling_factor)
                    elif self.cfg.optim.signed == "hard":
                        candidate.grad.sign_()
                    else:
                        pass

            self.current_task_loss = total_task_loss  # Side-effect this because of L-BFGS closure limitations :<
            return total_objective

        return closure

    def _score_trial(self, candidate, labels, rec_model, shared_data):
        """Score candidate solutions based on some criterion."""

        if self.cfg.restarts.scoring in ["euclidean", "cosine-similarity"]:
            objective = Euclidean() if self.cfg.restarts.scoring == "euclidean" else CosineSimilarity()
            objective.initialize(self.loss_fn, self.cfg.impl, shared_data[0]["metadata"]["local_hyperparams"])
            score = 0
            for model, data in zip(rec_model, shared_data):
                score += objective(model, data["gradients"], candidate, labels)[0]
        elif self.cfg.restarts.scoring in ["TV", "total-variation"]:
            score = TotalVariation(scale=1.0)(candidate)
        else:
            raise ValueError(f"Scoring mechanism {self.cfg.scoring} not implemented.")
        return score if score.isfinite() else float("inf")

    def _select_optimal_reconstruction(self, candidate_solutions, scores, stats):
        """Choose one of the candidate solutions based on their scores (for now).

        More complicated combinations are possible in the future."""
        optimal_val, optimal_index = torch.min(scores, dim=0)
        optimal_solution = candidate_solutions[optimal_index]
        stats["opt_value"] = optimal_val.item()
        if optimal_val.isfinite():
            log.info(f"Optimal candidate solution with rec. loss {optimal_val.item():2.4f} selected.")
            return optimal_solution
        else:
            log.info("No valid reconstruction could be found.")
            return torch.zeros_like(optimal_solution)

    def _init_t_optimizer(self):
        return torch.optim.Adam([self.t], lr=self.cfg.optim.eta_t)

    def _init_d_optimizer(self):
        return torch.optim.Adam(self.d.parameters(), lr=self.cfg.optim.eta_d)

    def _init_p1_optimizer(self):
        return torch.optim.Adam(self.P1.parameters(), lr=self.cfg.optim.eta_P)

    def _update_bezier_quadratic_(self, t, P1, w0, wT):
        return (1-t)**2 * w0 + 2 * (1-t) * t * P1 + t**2 * wT

    def _update_bezier_linear(self, t, P1, w0, wT):
        return (1-t) * w0 + t * wT