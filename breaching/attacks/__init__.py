"""Load attacker code and instantiate appropriate objects."""
import torch

from .optimization_based_attack import OptimizationBasedAttacker
from .optimization_based_april import OptimizationAprilAttacker
from .cocktail_party_attack import CocktailPartyAttacker
from .multiscale_optimization_attack import MultiScaleOptimizationAttacker
from .optimization_with_label_attack import OptimizationJointAttacker
from .optimization_permutation_attack import OptimizationPermutationAttacker
from .analytic_attack import AnalyticAttacker, ImprintAttacker, DecepticonAttacker, AprilAttacker
from .recursive_attack import RecursiveAttacker


def prepare_attack(model, loss, cfg_attack, setup=dict(dtype=torch.float, device=torch.device("cpu"))):
    if cfg_attack.attack_type == "optimization":
        attacker = OptimizationBasedAttacker(model, loss, cfg_attack, setup)
    elif cfg_attack.attack_type == "multiscale":
        attacker = MultiScaleOptimizationAttacker(model, loss, cfg_attack, setup)
    elif cfg_attack.attack_type == "april-optimization":
        attacker = OptimizationAprilAttacker(model, loss, cfg_attack, setup)
    elif cfg_attack.attack_type == "cocktail-party-optimization":
        attacker = CocktailPartyAttacker(model, loss, cfg_attack, setup)
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
    else:
        raise ValueError(f"Invalid type of attack {cfg_attack.attack_type} given.")

    return attacker


__all__ = ["prepare_attack"]
