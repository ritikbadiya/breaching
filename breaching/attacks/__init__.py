"""Load attacker code and instantiate appropriate objects."""
import torch

from .optimization_based_attack import OptimizationBasedAttacker
from .optimization_based_april import OptimizationAprilAttacker
from .multiscale_optimization_attack import MultiScaleOptimizationAttacker
from .optimization_with_label_attack import OptimizationJointAttacker
from .optimization_permutation_attack import OptimizationPermutationAttacker
from .analytic_attack import AnalyticAttacker, ImprintAttacker, DecepticonAttacker, AprilAttacker
from .recursive_attack import RecursiveAttacker
# from .gradvit import GradVit
from .nonlinear_surrogateme import NonLinearSurrogateModelExtension
from .boosting_gla import BoostingGLA
from .gan_gradmatching_based_attack import GANGradMatchingAttacker
from .girg_attack import ConditionalGANGradMatchingAttacker
from .cgir_attack import ConditionalGenInstRecAttacker
from .gismn_attack import GenerativeStyleMigrationAttacker
from .gias_attack import GenImagePriorAttacker
from .seer_attack import SEERAttacker  # NEW: Import SEER


def prepare_attack(model, loss, cfg_attack, setup=dict(dtype=torch.float, device=torch.device("cpu"))):
    if cfg_attack.attack_type == "optimization":
        attacker = OptimizationBasedAttacker(model, loss, cfg_attack, setup)
    elif cfg_attack.attack_type == "multiscale":
        attacker = MultiScaleOptimizationAttacker(model, loss, cfg_attack, setup)
    elif cfg_attack.attack_type == "april-optimization":
        attacker = OptimizationAprilAttacker(model, loss, cfg_attack, setup)
    elif cfg_attack.attack_type == "analytic":
        attacker = AnalyticAttacker(model, loss, cfg_attack, setup)
    elif cfg_attack.attack_type == "april-analytic":
        attacker = AprilAttacker(model, loss, cfg_attack, setup)
    elif cfg_attack.attack_type == "imprint-readout":
        attacker = ImprintAttacker(model, loss, cfg_attack, setup)
    elif cfg_attack.attack_type == "decepticon-readout":
        attacker = DecepticonAttacker(model, loss, cfg_attack, setup)
    elif cfg_attack.attack_type == "recursive":
        attacker = RecursiveAttacker(model, loss, cfg_attack, setup)
    elif cfg_attack.attack_type == "joint-optimization":
        attacker = OptimizationJointAttacker(model, loss, cfg_attack, setup)
    elif cfg_attack.attack_type == "permutation-optimization":
        attacker = OptimizationPermutationAttacker(model, loss, cfg_attack, setup)
    # elif cfg_attack.attack_type == "gradvit":
    #     attacker = GradVit(model, loss, cfg_attack, setup)
    elif cfg_attack.attack_type == "nl-sme":
        attacker = NonLinearSurrogateModelExtension(model, loss, cfg_attack, setup)
    elif cfg_attack.attack_type == "boosting-gla":
        attacker = BoostingGLA(model, loss, cfg_attack, setup)
    elif cfg_attack.attack_type == "cinet":
        attacker = GANGradMatchingAttacker(model, loss, cfg_attack, setup)
    elif cfg_attack.attack_type == "girg":
        attacker = ConditionalGANGradMatchingAttacker(model, loss, cfg_attack, setup)
    elif cfg_attack.attack_type == "cgir":
        attacker = ConditionalGenInstRecAttacker(model, loss, cfg_attack, setup)
    elif cfg_attack.attack_type == "gismn":
        attacker = GenerativeStyleMigrationAttacker(model, loss, cfg_attack, setup)
    elif cfg_attack.attack_type == "gias":
        attacker = GenImagePriorAttacker(model, loss, cfg_attack, setup)
    elif cfg_attack.attack_type == "seer":
        attacker = SEERAttacker(model, loss, cfg_attack, setup)
    else:
        raise ValueError(f"Invalid type of attack {cfg_attack.attack_type} given.")

    return attacker


__all__ = ["prepare_attack"]
