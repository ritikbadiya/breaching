"""Load attacker code and instantiate appropriate objects."""
import importlib
import torch
import logging
log = logging.getLogger(__name__)
# Lazily import attack implementations to avoid importing heavy optional
# dependencies (e.g., transformers/keras/pyarrow) unless they are needed.
_ATTACK_REGISTRY = {
    "optimization": (".optimization_based_attack", "OptimizationBasedAttacker"),
    "multiscale": (".multiscale_optimization_attack", "MultiScaleOptimizationAttacker"),
    "april-optimization": (".optimization_based_april", "OptimizationAprilAttacker"),
    "analytic": (".analytic_attack", "AnalyticAttacker"),
    "april-analytic": (".analytic_attack", "AprilAttacker"),
    "imprint-readout": (".analytic_attack", "ImprintAttacker"),
    "decepticon-readout": (".analytic_attack", "DecepticonAttacker"),
    "recursive": (".recursive_attack", "RecursiveAttacker"),
    "joint-optimization": (".optimization_with_label_attack", "OptimizationJointAttacker"),
    "permutation-optimization": (".optimization_permutation_attack", "OptimizationPermutationAttacker"),
    "nl-sme": (".nonlinear_surrogateme", "NonLinearSurrogateModelExtension"),
    "boosting-gla": (".boosting_gla", "BoostingGLA"),
    "cinet": (".gan_gradmatching_based_attack", "GANGradMatchingAttacker"),
    "girg": (".girg_attack", "ConditionalGANGradMatchingAttacker"),
    "cgir": (".cgir_attack", "ConditionalGenInstRecAttacker"),
    "gismn": (".gismn_attack", "GenerativeStyleMigrationAttacker"),
    "gias": (".gias_attack", "GenImagePriorAttacker"),
    "ss-peft": (".gismn_peft_domainalign", "SemanticSimilarityBasedLatentCodeAttackerPEFT"),
}


def _load_attacker(attack_type: str):
    entry = _ATTACK_REGISTRY.get(attack_type)
    if entry is None:
        raise ValueError(f"Invalid type of attack {attack_type} given.")
    module_name, class_name = entry
    module = importlib.import_module(module_name, package=__name__)
    return getattr(module, class_name)

def prepare_attack(model, loss, cfg_attack, setup=dict(dtype=torch.float, device=torch.device("cpu")), **kwargs):
    # log.warning(f"PEFT Configuration: {kwargs.get('cfg_peft', None)}")
    attacker_cls = _load_attacker(cfg_attack.attack_type)
    attacker = attacker_cls(model, loss, cfg_attack, setup, **kwargs)
    return attacker


__all__ = ["prepare_attack"]
