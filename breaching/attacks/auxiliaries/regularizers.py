"""Various regularizers that can be re-used for multiple attacks."""

import torch
import torchvision
from .deepinversion import DeepInversionFeatureHook

MOCOV2_RESNET50_URL = "https://dl.fbaipublicfiles.com/moco/moco_checkpoints/moco_v2_800ep/moco_v2_800ep_pretrain.pth.tar"

class _LinearFeatureHook:
    """Hook to retrieve input to given module."""

    def __init__(self, module):
        self.features = None
        self.hook = module.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        input_features = input[0]
        self.features = input_features

    def close(self):
        self.hook.remove()


class FeatureRegularization(torch.nn.Module):
    """Feature regularization implemented for the last linear layer at the end."""

    def __init__(self, setup, scale=0.1):
        super().__init__()
        self.setup = setup
        self.scale = scale

    def initialize(self, models, shared_data, labels, *args, **kwargs):
        self.measured_features = []
        for user_data in shared_data:
            # Assume last two gradient vector entries are weight and bias:
            weights = user_data["gradients"][-2]
            bias = user_data["gradients"][-1]
            grads_fc_debiased = weights / bias[:, None]
            features_per_label = []
            for label in labels:
                if bias[label] != 0:
                    features_per_label.append(grads_fc_debiased[label])
                else:
                    features_per_label.append(torch.zeros_like(grads_fc_debiased[0]))
            self.measured_features.append(torch.stack(features_per_label))

        self.refs = [None for model in models]
        for idx, model in enumerate(models):
            for module in model.modules():
                # Keep only the last linear layer here:
                if isinstance(module, torch.nn.Linear):
                    self.refs[idx] = _LinearFeatureHook(module)

    def forward(self, tensor, *args, **kwargs):
        regularization_value = 0
        for ref, measured_val in zip(self.refs, self.measured_features):
            regularization_value += (ref.features - measured_val).pow(2).mean()
        return regularization_value * self.scale

    def __repr__(self):
        return f"Feature space regularization, scale={self.scale}"


class LinearLayerRegularization(torch.nn.Module):
    """Linear layer regularization implemented for arbitrary linear layers. WIP Example."""

    def __init__(self, setup, scale=0.1):
        super().__init__()
        self.setup = setup
        self.scale = scale

    def initialize(self, models, gradient_data, *args, **kwargs):
        self.measured_features = []
        self.refs = [list() for model in models]

        for idx, (model, user_data) in enumerate(zip(models, gradient_data)):
            # 1) Find linear layers:
            linear_layers = []
            for name, module in model.named_modules():
                if isinstance(module, torch.nn.Linear):
                    linear_layers.append(name)
                    self.refs[idx].append(_LinearFeatureHook(module))
            named_grads = {name: g for (g, (name, param)) in zip(user_data["gradients"], model.named_parameters())}

            # 2) Check features
            features = []
            for linear_layer in linear_layers:
                weights = named_grads[linear_layer + ".weight"]
                bias = named_grads[linear_layer + ".bias"]
                grads_fc_debiased = (weights / bias[:, None]).mean(dim=0)  # At some point todo: Make this smarter
                features.append(grads_fc_debiased)
            self.measured_features.append(features)

    def forward(self, tensor, *args, **kwargs):
        regularization_value = 0
        for model_ref, data_ref in zip(self.refs, self.measured_features):
            for linear_layer, data in zip(model_ref, data_ref):
                regularization_value += (linear_layer.features.mean(dim=0) - data).pow(2).sum()

    def __repr__(self):
        return f"Feature space regularization, scale={self.scale}"


class TotalVariation(torch.nn.Module):
    """Computes the total variation value of an (image) tensor, based on its last two dimensions.
    Optionally also Color TV based on its last three dimensions.

    The value of this regularization is scaled by 1/sqrt(M*N) times the given scale."""

    def __init__(self, setup, scale=0.1, inner_exp=1, outer_exp=1, double_opponents=False, eps=1e-8):
        """scale is the overall scaling. inner_exp and outer_exp control isotropy vs anisotropy.
        Optionally also includes proper color TV via double opponents."""
        super().__init__()
        self.setup = setup
        self.scale = scale
        self.inner_exp = inner_exp
        self.outer_exp = outer_exp
        self.eps = eps
        self.double_opponents = double_opponents

    def initialize(self, models, *args, **kwargs):
        # We assume all shared data and models have the same number of channels
        # This is a bit of a hack to get the channel count here if it's not passed to init
        # In this framework, tensor.shape[1] at forward time is the ultimate source of truth.
        pass

    def forward(self, tensor, *args, **kwargs):
        """Use a convolution-based approach."""
        channels = tensor.shape[1]
        if self.double_opponents and channels == 3:
            tensor = torch.cat(
                [
                    tensor,
                    tensor[:, 0:1, :, :] - tensor[:, 1:2, :, :],
                    tensor[:, 0:1, :, :] - tensor[:, 2:3, :, :],
                    tensor[:, 1:2, :, :] - tensor[:, 2:3, :, :],
                ],
                dim=1,
            )
            groups = 6
        else:
            groups = channels

        # Construct weight on the fly if needed or cache it based on channels
        if not hasattr(self, "_weight") or self._weight.shape[1] * self._weight.shape[0] // 2 != channels:
            grad_weight = torch.tensor([[0, 0, 0], [0, -1, 1], [0, 0, 0]], **self.setup).unsqueeze(0).unsqueeze(1)
            grad_weight = torch.cat((torch.transpose(grad_weight, 2, 3), grad_weight), 0)
            grad_weight = torch.cat([grad_weight] * groups, 0)
            self.register_buffer("_weight", grad_weight, persistent=False)

        diffs = torch.nn.functional.conv2d(
            tensor, self._weight, None, stride=1, padding=1, dilation=1, groups=groups
        )
        squares = (diffs.abs() + self.eps).pow(self.inner_exp)
        squared_sums = (squares[:, 0::2] + squares[:, 1::2]).pow(self.outer_exp)
        return squared_sums.mean() * self.scale

    def __repr__(self):
        return (
            f"Total Variation, scale={self.scale}. p={self.inner_exp} q={self.outer_exp}. "
            f"{'Color TV: double oppponents' if self.double_opponents else ''}"
        )


class OrthogonalityRegularization(torch.nn.Module):
    """This is the orthogonality regularizer described Qian et al.,

    "MINIMAL CONDITIONS ANALYSIS OF GRADIENT-BASED RECONSTRUCTION IN FEDERATED LEARNING"
    """

    def __init__(self, setup, scale=0.1):
        super().__init__()
        self.setup = setup
        self.scale = scale

    def initialize(self, models, *args, **kwargs):
        pass

    def forward(self, tensor, *args, **kwargs):
        if tensor.shape[0] == 1:
            return 0
        else:
            B = tensor.shape[0]
            full_products = (tensor.unsqueeze(0) * tensor.unsqueeze(1)).pow(2).view(B, B, -1).mean(dim=2)
            idx = torch.arange(0, B, out=torch.LongTensor())
            full_products[idx, idx] = 0
            return full_products.sum()

    def __repr__(self):
        return f"Input Orthogonality, scale={self.scale}"


class NormRegularization(torch.nn.Module):
    """Implement basic norm-based regularization, e.g. an L2 penalty."""

    def __init__(self, setup, scale=0.1, pnorm=2.0):
        super().__init__()
        self.setup = setup
        self.scale = scale
        self.pnorm = pnorm

    def initialize(self, models, *args, **kwargs):
        pass

    def forward(self, tensor, *args, **kwargs):
        return 1 / self.pnorm * tensor.pow(self.pnorm).mean() * self.scale

    def __repr__(self):
        return f"Input L^p norm regularization, scale={self.scale}, p={self.pnorm}"


class DeepInversion(torch.nn.Module):
    """Implement a DeepInversion based regularization as proposed in DeepInversion as used for reconstruction in
    Yin et al, "See through Gradients: Image Batch Recovery via GradInversion".
    """

    def __init__(self, setup, scale=0.1, first_bn_multiplier=10):
        super().__init__()
        self.setup = setup
        self.scale = scale
        self.first_bn_multiplier = first_bn_multiplier

    def initialize(self, models, *args, **kwargs):
        """Initialize forward hooks."""
        self.losses = [list() for model in models]
        for idx, model in enumerate(models):
            for module in model.modules():
                if isinstance(module, torch.nn.BatchNorm2d):
                    self.losses[idx].append(DeepInversionFeatureHook(module))

    def forward(self, tensor, *args, **kwargs):
        rescale = [self.first_bn_multiplier] + [1.0 for _ in range(len(self.losses[0]) - 1)]
        feature_reg = 0
        for loss in self.losses:
            feature_reg += sum([mod.r_feature * rescale[idx] for (idx, mod) in enumerate(loss)])
        return self.scale * feature_reg

    def __repr__(self):
        return f"Deep Inversion Regularization (matching batch norms), scale={self.scale}, first-bn-mult={self.first_bn_multiplier}"

class ImagePrior(torch.nn.Module):
    """Implements the ImagePrior regularization as used in Grad-ViT. 
    Compare the BN stats with MoCoV2 pretrained ResNet50 features.
    Different from the one implemented in DeepInversion and STG."""
    def __init__(self, setup, scale=0.1, first_bn_multiplier=10, max_iters = 120000):
        super().__init__()
        self.setup = setup
        self.scale = scale
        self.step = 0
        self.max_iters = max_iters
        self.first_bn_multiplier = first_bn_multiplier
        # self.moco = torch.hub.load('facebookresearch/moco:main', 'moco_v2_resnet50', force_reload=True)
        moco_state_dict = torch.hub.load_state_dict_from_url(MOCOV2_RESNET50_URL, map_location='cpu', progress=True)['state_dict']
        self.moco = torchvision.models.resnet50(pretrained = False)
        self.moco.load_state_dict({k.replace('module.encoder_q.',''): v for k, v in moco_state_dict.items() if k.startswith('module.encoder_q') and not k.startswith('module.encoder_q.fc')}, strict = False)
        self.moco.to(setup['device'])
        self.moco.eval()

    def initialize(self, models, *args, **kwargs):
        """Initialize forward hooks."""
        self.losses = []
    
        for module in self.moco.modules():
            if isinstance(module, torch.nn.BatchNorm2d):
                self.losses.append(DeepInversionFeatureHook(module))
    
    def forward(self, tensor, *args, **kwargs):
        if self.step < self.max_iters//2:
            self.step += 1
            return 0.0
        else:
            self.step += 1
            #### To handle MNIST-like images ####
            B, C, H, W = tensor.shape
            if C==1:
                tensor = tensor.repeat(1,3,1,1)
            if H < 32 and W < 32:
                tensor = torch.nn.functional.interpolate(tensor, size=(32,32), mode='bilinear', align_corners=False)

            _ = self.moco(tensor)
            rescale = [self.first_bn_multiplier] + [1.0 for _ in range(len(self.losses) - 1)]
            feature_reg = sum([mod.r_feature * rescale[idx] for (idx, mod) in enumerate(self.losses)])
            return self.scale * feature_reg

    def __repr__(self):
        return f"Image Prior Regularization with MoCoV2 pretrained ResNet50, scale={self.scale}"
    
class PatchPrior(torch.nn.Module):
    """Implements the PatchPrior regularization as used in Grad-ViT. 
    Compute Total Variation on patches of the image instead of the whole image.
    Pixel values among adjacent patch edges shall be in similar ranges."""
    def __init__(self, setup, scale=0.1, inner_exp=1, outer_exp=1, double_opponents=False, eps=1e-8, patch_size=14):
        super().__init__()
        self.patch_size = patch_size
        self.setup = setup
        self.scale = scale
        self.inner_exp = inner_exp
        self.outer_exp = outer_exp
        self.eps = eps
        self.double_opponents = double_opponents

    def initialize(self, models, *args, **kwargs):
        pass

    def total_variation_patches(self, x, P=16):
        """Patch-based anisotropic TV computed on patch boundaries.

        For a patch size `P`, compute the L2 norm of the difference across
        vertical patch boundaries (between rows P*k-1 and P*k) and horizontal
        patch boundaries (between cols P*k-1 and P*k). For each boundary the
        difference is flattened across channels and the remaining spatial
        dimension, producing a per-sample L2 norm. The function returns the
        mean (over the batch) of the summed per-sample norms across all
        boundaries.

        Args:
            x (torch.Tensor): input tensor with shape (B, C, H, W).
            P (int): patch size (default 16).

        Returns:
            torch.Tensor: scalar tensor containing the mean patch-TV penalty.
        """
        B, C, H, W = x.shape

        # Prepare per-sample accumulator
        per_sample = x.new_zeros((B,))

        # Vertical boundaries (along height): compare rows P*k-1 and P*k
        n_vert = int(H) // P
        for k in range(1, n_vert):
            a = x[:, :, P * k, :].reshape(B, -1)       # shape (B, C*W)
            b = x[:, :, P * k - 1, :].reshape(B, -1)   # shape (B, C*W)
            diff = a - b
            norms = torch.norm(diff, p=2, dim=1)      # per-sample L2
            per_sample = per_sample + norms

        # Horizontal boundaries (along width): compare cols P*k-1 and P*k
        n_horiz = int(W) // P
        for k in range(1, n_horiz):
            a = x[:, :, :, P * k].reshape(B, -1)      # shape (B, C*H)
            b = x[:, :, :, P * k - 1].reshape(B, -1)  # shape (B, C*H)
            diff = a - b
            norms = torch.norm(diff, p=2, dim=1)      # per-sample L2
            per_sample = per_sample + norms

        # If there were no boundaries (P larger than H or W), return zero
        if (n_vert <= 1) and (n_horiz <= 1):
            return x.new_tensor(0.0)

        # Return mean over batch of summed per-sample boundary norms
        return per_sample.mean()

    def forward(self, tensor, *args, **kwargs):
        """Compute Total Variation on patches of the image."""
        tv_loss = self.total_variation_patches(tensor, P=self.patch_size)
        return tv_loss

    def __repr__(self):
        return f"Patch Prior Total Variation, scale={self.scale}, patch_size={self.patch_size}, p={self.inner_exp} q={self.outer_exp}. {'Color TV: double oppponents' if self.double_opponents else ''}"

class GroupRegularization(torch.nn.Module):
    """Group regularization placeholder - not implemented."""

    def __init__(self, setup, scale=0.1, aligner=None, x_list=None, warmup_iters=0):
        super().__init__()
        self.setup = setup
        self.scale = scale
        self.aligner = aligner
        self.x_list = x_list
        self.warmup_iters = warmup_iters
        self.iter = 0

    def compute_mean(self, x_list):
    # x_list: list of tensors, each (B, C, H, W)
        return torch.stack(x_list, dim=0).mean(dim=0)

    def initialize(self, models, *args, **kwargs):
        self.iter = 0

    @torch.no_grad()
    def compute_consensus(self, x_list):
        """
        Compute final consensus x_C by:
        1) computing mean x_m
        2) aligning each x_s to x_m
        3) averaging aligned results
        """

        x_m = self.compute_mean(x_list)

        aligned = []
        for x_s in x_list:
            x_s_aligned = self.aligner(x_s, x_m)
            aligned.append(x_s_aligned)

        x_C = torch.stack(aligned, dim=0).mean(dim=0)
        return x_C

    def forward(self, tensor, *args, **kwargs):
        """
        tensor: current reconstruction x̂
        expects:
            self.x_list = list of all candidate reconstructions
        """
        self.iter += 1

        # warmup: don't apply consensus too early
        if self.iter < self.warmup_iters:
            return tensor.new_tensor(0.0)

        if self.x_list is None or len(self.x_list) < 2:
            return tensor.new_tensor(0.0)

        # compute consensus (no gradients through alignment)
        with torch.no_grad():
            x_C = self.compute_consensus(self.x_list)

        # L2 penalty to consensus
        reg = torch.mean((tensor - x_C)**2)
        return self.scale*reg

    def __repr__(self):
        return (
            f"Group (Consensus) Regularization via RANSAC-Flow, "
            f"scale={self.scale}, warmup={self.warmup_iters}"
        )


regularizer_lookup = dict(
    total_variation=TotalVariation,
    orthogonality=OrthogonalityRegularization,
    norm=NormRegularization,
    deep_inversion=DeepInversion,
    features=FeatureRegularization,
    image_prior=ImagePrior,
    patch_prior=PatchPrior,
    group_regularization=GroupRegularization
)
