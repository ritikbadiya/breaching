"""Various regularizers that can be re-used for multiple attacks."""

import torch
import torchvision
from .deepinversion import DeepInversionFeatureHook
import logging
import kornia as K

log = logging.getLogger(__name__)

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

class L2Regularization(torch.nn.Module):
    """L2 regularization."""

    def __init__(self, setup, scale=0.1):
        super().__init__()
        self.setup = setup
        self.scale = scale
    
    def initialize(self, models, shared_data, labels, *args, **kwargs):
        pass

    def forward(self, tensor, *args, **kwargs):
        if kwargs.get('target', None) is not None:
            if isinstance(tensor, list) and isinstance(kwargs['target'], list):
                loss = 0
                for t, tt in zip(tensor, kwargs['target']):
                    loss += torch.pow(t - tt, 2).mean()
                return loss * self.scale
            elif isinstance(tensor, list) and not isinstance(kwargs['target'], list):
                loss = 0
                for t in tensor:
                    loss += torch.pow(t - kwargs['target'], 2).mean()
                return loss * self.scale
            return torch.pow(tensor - kwargs['target'], 2).mean() * self.scale
        return torch.pow(tensor, 2).mean() * self.scale

class LabelRegularization(L2Regularization):
    """L2 regularization on the label embedding space."""

    def __init__(self, setup, scale=0.1):
        super().__init__(setup, scale)


class L1Regularization(torch.nn.Module):
    """L1 regularization implemented for the last linear layer at the end."""

    def __init__(self, setup, scale=0.1):
        super().__init__()
        self.setup = setup
        self.scale = scale
    
    def initialize(self, models, shared_data, labels, *args, **kwargs):
        pass

    def forward(self, tensor, *args, **kwargs):
        return torch.abs(tensor).mean() * self.scale

class SignRegularization(torch.nn.Module):
    """Sign regularization implemented for the last linear layer at the end."""

    def __init__(self, setup, scale=0.1):
        super().__init__()
        self.setup = setup
        self.scale = scale
    
    def initialize(self, models, shared_data, labels, *args, **kwargs):
        pass

    def forward(self, tensor, *args, **kwargs):
        return torch.mean(torch.minimum(torch.relu(tensor), torch.relu(-tensor))) * self.scale

class MIRegularization(torch.nn.Module):
    """Mutual information regularization implemented for the last linear layer at the end."""

    def __init__(self, setup, scale=2.0):
        super().__init__()
        self.setup = setup
        self.scale = scale
    
    def initialize(self, models, shared_data, labels, *args, **kwargs):
        pass

    def forward(self, tensor, *args, **kwargs):
        U_norm = torch.nn.functional.normalize(tensor, dim=1)
        cos = U_norm @ U_norm.T
        # mask = ~torch.eye(tensor.shape[0], dtype=torch.bool, device=tensor.device)
        return self.scale * torch.mean(torch.exp(2.0 * torch.abs(cos)))

    def __repr__(self):
        return f"Mutual information regularization (Cocktail Party Attack), scale={self.scale}"        

class ICAFeatureRegularization(torch.nn.Module):
    """Feature regularization implemented for the last linear layer at the end."""

    def __init__(self, setup, scale=0.1, mi_scale=1e-3, sign_scale=1e-3, l1_scale=1e-3):
        super().__init__()
        self.setup = setup
        self.scale = scale
        self.sign_reg = SignRegularization(setup, sign_scale)
        self.mi_reg = MIRegularization(setup, mi_scale)
        self.sparsity_reg = L1Regularization(setup, l1_scale)

    def center_and_whiten(self, G, eps=1e-5):
        """
        G: (n, d) gradient matrix
        """
        G = G - G.mean(dim=1, keepdim=True)
        cov = (G @ G.T) / G.shape[1]
        eigvals, eigvecs = torch.linalg.eigh(cov)
        W = eigvecs @ torch.diag(1.0 / torch.sqrt(eigvals + eps)) @ eigvecs.T
        return W @ G

    def safe_negentropy(self, z):
        return torch.mean(torch.abs(z) + torch.log1p(torch.exp(-2 * torch.abs(z))) - torch.log(torch.tensor(2.0, device=z.device)))
    
    def negentropy(self, z, a=2.0):
        return torch.mean(torch.log(torch.cosh(a * z).square() + 1e-5) / a**2)

    def cpa_loss(self, U, G, *args, **kwargs):
        """
        U: (B, n)
        G: (n, d)
        """
        Z = U @ G
        Z_ = Z / (Z.std(dim=1, keepdim=True) + 1e-6)
        J = torch.mean(torch.stack([self.safe_negentropy(Z_[i]) for i in range(Z_.shape[0])]))
        SP = self.sparsity_reg(Z)
        SR = self.sign_reg(Z)
        MI = self.mi_reg(U)
        # log.info(f"J: {J.item():2.4f}, SP: {SP.item():2.4f}, SR: {SR.item():2.4f}, MI: {MI.item():2.4f}")
        return -(J-SP-SR-MI)

    def initialize(self, models, shared_data, labels, *args, **kwargs):
        self.refs = [None for model in models]
        for idx, model in enumerate(models):
            for name, module in model.named_modules():
                # Keep only the last linear layer here:
                # enumerating over name essential for ViT as the attention layers are also nn.Linear
                if isinstance(module, torch.nn.Linear) and 'head' in name:
                    self.refs[idx] = _LinearFeatureHook(module)

        self.measured_features = []
        for user_data in shared_data:
            # Assume last two gradient vector entries are weight and bias:
            G = user_data["gradients"][-2]
            G = self.center_and_whiten(G) # footnote Page no. 4 CPA Paper
            U = torch.randn(labels.shape[0], G.shape[0], requires_grad=True, device=G.device)
            optimizer = torch.optim.LBFGS([U], lr=1.0, max_iter=10, line_search_fn="strong_wolfe")
            def closure():
                optimizer.zero_grad()
                loss = self.cpa_loss(U, G)
                # log.info(f"CPA Loss: {loss.item():2.4f}")
                loss.backward()
                return loss
            for _ in range(10):
                optimizer.step(closure)
                # with torch.no_grad():
                #     Uu, _, Vh = torch.linalg.svd(U, full_matrices=False)
                #     U.copy_(Uu @ Vh) # project U back to the Stiefel manifold
                    # U.copy_(torch.nn.functional.normalize(U, dim=1))
                # enforces U.U^T = I
            Z_hat = U @ G   # recovered private embeddings
            Z_hat = torch.where(Z_hat.mean(dim=1, keepdim=True) < 0, -Z_hat, Z_hat)
            self.measured_features.append(Z_hat)

    def forward(self, tensor, *args, **kwargs):
        regularization_value = 0
        for ref, measured_val in zip(self.refs, self.measured_features):
            # log.info(f"Shape of ref.features: {ref.features.shape}")
            # log.info(f"Shape of measured_val: {measured_val.shape}")
            regularization_value += torch.nn.functional.mse_loss(ref.features, measured_val)
        return regularization_value * self.scale

    def __repr__(self):
        return f"ICA feature space regularization (Cocktail Party Attack), scale={self.scale}"        

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
            for name, module in model.named_modules():
                # Keep only the last linear layer here:
                # enumerating over name essential for ViT as the attention layers are also nn.Linear
                if isinstance(module, torch.nn.Linear) and 'head' in name:
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

class MeanRegularization(torch.nn.Module):
    def __init__(self, setup, scale=0.1):
        super().__init__()
        self.setup = setup # for putting things on right device
        self.scale = scale
        self.register_buffer(
            "means",
            torch.tensor([0.491, 0.467, 0.421])
        )
    
    def initialize(self, models, shared_data, labels, *args, **kwargs):
        pass

    def forward(self, x):
        # x: (B, C, H, W)
        assert x.shape[1] == 3, "MeanRegularization expects 3-channel images"

        x_means = x.mean(dim=(0, 2, 3))  # (C,)
        return self.scale*torch.mean((x_means - self.means)**2)
    
    def __repr__(self):
        return f"Mean Regularization, scale={self.scale}"

class CannyEdgeRegularization(torch.nn.Module):
    def __init__(self, setup, beta=0.4, theta1=0.1, theta2=0.2, scale=0.1):
        super().__init__()
        self.setup = setup # for putting things on right device
        self.scale = scale
        self.beta = beta
        self.theta1 = theta1
        self.theta2 = theta2
        self.register_buffer(
            "grad_edge_center",
            torch.zeros(2)
        )

    def initialize(self, models, shared_data, labels, *args, **kwargs):
        user_data = shared_data[0]
        G_fc = user_data["gradients"][-2]
        G = G_fc.abs().mean(dim=0)
        H = W = int(G.numel() ** 0.5)
        g = G.flatten()
        f_in = (g.max() - g.mean()) * self.beta
        idx = torch.nonzero(g > f_in, as_tuple=False).squeeze(1)

        if idx.numel() == 0:
            # fallback: center of image
            self.grad_edge_center = torch.tensor(
                [H // 2, W // 2], device=G.device
            )
            return
        
        rows = idx // W
        cols = idx % W

        coords = torch.stack([rows, cols], dim=1)
        self.grad_edge_center = coords.median(dim=0).values

    def forward(self, x):
        B, C, H, W = x.shape
        if C == 3:
            gray = K.color.rgb_to_grayscale(x)
        else:
            gray = x

        edges, _ = K.filters.canny(
            gray,
            low_threshold=self.theta1,
            high_threshold=self.theta2,
        )

        coords = torch.nonzero(edges[0, 0] > 0, as_tuple=False)

        if coords.numel() == 0:
            return x.new_tensor(0.0)

        img_edge_center = coords[coords.shape[0] // 2].float()
        Red = torch.norm(self.grad_edge_center - img_edge_center, p=2)

        return self.scale*Red
    
    def __repr__(self):
        return (
            f"Canny Edge Regularization "
            f"(beta={self.beta}, scale={self.scale})"
        )

regularizer_lookup = dict(
    total_variation=TotalVariation,
    orthogonality=OrthogonalityRegularization,
    norm=NormRegularization,
    deep_inversion=DeepInversion,
    features=FeatureRegularization,
    ica_features=ICAFeatureRegularization,
    image_prior=ImagePrior,
    patch_prior=PatchPrior,
    group_regularization=GroupRegularization,
    sign_regularization=SignRegularization,
    mi_regularization=MIRegularization,
    l2_regularization=L2Regularization,
    l2_regularization_target=L2Regularization,
    label_regularization=LabelRegularization,
    l1_regularization=L1Regularization,
    mean_regularization=MeanRegularization, 
    canny_edge_regularization=CannyEdgeRegularization
)
