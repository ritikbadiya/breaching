"""SEER Attack: Hiding in Plain Sight - Disguising Data Stealing Attacks in Federated Learning

Paper: "Hiding in Plain Sight: Disguising Data Stealing Attacks in Federated Learning" (ICLR 2024)
Authors: INSAIT Institute
GitHub: https://github.com/insait-institute/SEER
"""

import torch
import torch.nn as nn
import logging
from collections import defaultdict
import copy

from .base_attack import _BaseAttacker
from .auxiliaries.common import optimizer_lookup

log = logging.getLogger(__name__)


class SEERAttacker(_BaseAttacker):
    """SEER Attack Implementation.
    
    SEER combines three key components:
    1. ParamSelector: Selects sparse gradients for efficiency
    2. Disaggregator: Encodes gradients to hidden representation
    3. Reconstructor: Decodes to reconstruct original data
    
    All trained jointly in an end-to-end manner.
    """

    def __init__(self, model, loss_fn, cfg_attack, setup=dict(dtype=torch.float, device=torch.device("cpu"))):
        super().__init__(model, loss_fn, cfg_attack, setup)
        self.cfg = cfg_attack
        self.device = setup["device"]
        self.dtype = setup["dtype"]

    def __repr__(self):
        return f"""Attacker (of type {self.__class__.__name__}) with settings:
    Parameter Selection:
        - Fraction: {self.cfg.param_selection.frac}
        - Size: {self.cfg.param_selection.size}
        - Seed: {self.cfg.param_selection.seed}
    Disaggregator (Encoder):
        - Hidden fraction: {self.cfg.disaggregator.mid_rep_frac}
    Reconstructor (Decoder):
        - Type: {self.cfg.reconstructor.type}
    Optimization:
        - Optimizer: {self.cfg.optim.optimizer}
        - Learning rate: {self.cfg.optim.step_size}
        - Max iterations: {self.cfg.optim.max_iterations}
        - Gradient clip: {self.cfg.optim.grad_clip}
        """

    def reconstruct(self, server_payload, shared_data, server_secrets=None, dryrun=False):
        """Main SEER reconstruction attack.
        
        Args:
            server_payload: List[dict] - Model parameters and metadata from server
            shared_data: List[dict] - Gradient data from clients
            server_secrets: Optional dict for additional attacks
            dryrun: bool - If True, run minimal iterations for testing
            
        Returns:
            reconstructed_data: Dict with 'data' and 'labels' tensors
            stats: Dict with training statistics
        """
        # Step 1: Prepare attack (inherited from _BaseAttacker)
        rec_models, labels, stats = self.prepare_attack(server_payload, shared_data)
        
        # Step 2: Build SEER networks
        seer_module = self._build_seer_networks(rec_models[0], shared_data[0])
        
        # Step 3: Setup optimizer
        optimizer, scheduler = optimizer_lookup(
            params=seer_module.parameters(),
            optim_name=self.cfg.optim.optimizer,
            step_size=self.cfg.optim.step_size,
            scheduler=self.cfg.optim.step_size_decay,
            max_iterations=self.cfg.optim.max_iterations
        )
        
        # Step 4: Main optimization loop
        num_iterations = 1 if dryrun else self.cfg.optim.max_iterations
        
        for iteration in range(num_iterations):
            # Forward pass through SEER
            reconstructed_batch = self._seer_step(
                seer_module, rec_models, shared_data, labels, optimizer, iteration, stats
            )
            
            # Log progress
            if self.cfg.optim.callback > 0 and iteration % self.cfg.optim.callback == 0:
                log.info(f"SEER Iteration {iteration}/{num_iterations}, Loss: {stats['loss'][-1]:.6f}")

            if scheduler:
                scheduler.step()
        
        # Step 5: Prepare final output
        reconstructed_data = dict(data=reconstructed_batch, labels=labels)
        
        return reconstructed_data, stats

    def _build_seer_networks(self, model, user_data):
        """Build the three SEER components: ParamSelector, Disaggregator, Reconstructor.
        
        Args:
            model: The victim model
            user_data: Sample user data for dimension inference
            
        Returns:
            SEERModule: Combined attack network
        """
        # Component 1: Parameter Selector
        param_selector = ParamSelector(
            model=model,
            sz=self.cfg.param_selection.size,
            frac=self.cfg.param_selection.frac,
            seed=self.cfg.param_selection.seed,
            device=self.device
        )
        
        num_selected_params = param_selector.num_par
        log.info(f"Selected {num_selected_params} parameters out of model")
        
        # Component 2: Disaggregator (Encoder) - Compress gradients
        hidden_dim = int(
            self.data_shape[0] * self.data_shape[1] * self.data_shape[2] * 
            self.cfg.disaggregator.mid_rep_frac
        )
        
        disaggregator = nn.Linear(
            num_selected_params, 
            hidden_dim, 
            bias=False
        ).to(dtype=self.dtype, device=self.device)
        
        # Component 3: Reconstructor (Decoder)
        image_pixels = self.data_shape[0] * self.data_shape[1] * self.data_shape[2]
        
        if self.cfg.reconstructor.type == 'linear':
            reconstructor = nn.Linear(
                hidden_dim, 
                image_pixels, 
                bias=True
            ).to(dtype=self.dtype, device=self.device)
            
        elif self.cfg.reconstructor.type == 'deconv':
            reconstructor = DeconvDecoder(
                input_dim=hidden_dim,
                output_shape=self.data_shape,
                device=self.device,
                dtype=self.dtype
            )
        else:
            reconstructor = nn.Linear(
                hidden_dim, 
                image_pixels, 
                bias=True
            ).to(dtype=self.dtype, device=self.device)
        
        # Combine all components
        seer_net = SEERModule(param_selector, disaggregator, reconstructor)
        return seer_net

    def _seer_step(self, seer_module, rec_models, shared_data, labels, 
                   optimizer, iteration, stats):
        """Single SEER optimization step.
        
        Args:
            seer_module: The SEER attack network
            rec_models: List of model copies for gradient computation
            shared_data: Client gradient data
            labels: True labels
            optimizer: PyTorch optimizer
            iteration: Current iteration number
            stats: Statistics dict to track
            
        Returns:
            reconstructed_data: Reconstructed images (batch)
        """
        seer_module.train()
        optimizer.zero_grad()
        
        # Extract gradients from shared_data
        # shared_data[0]['gradients'] contains per-sample gradients
        batch_gradients = shared_data[0]["gradients"]
        
        # Forward pass through SEER
        reconstructed = seer_module(batch_gradients)
        
        # Reshape to image format
        batch_size = batch_gradients[0].shape[0] if isinstance(batch_gradients, (list, tuple)) else 1
        reconstructed_images = reconstructed.view(batch_size, *self.data_shape)
        
        # Compute loss
        loss = 0
        
        # Loss 1: Gradient matching loss
        if self.cfg.loss.gradient_loss_weight > 0:
            reconstructed_grads = self._compute_gradients(
                rec_models[0], reconstructed_images, labels
            )
            grad_loss = self._gradient_matching_loss(batch_gradients, reconstructed_grads)
            loss += self.cfg.loss.gradient_loss_weight * grad_loss
        
        # Loss 2: TV regularization
        if self.cfg.loss.tv_loss_weight > 0:
            tv_loss = self._total_variation_loss(reconstructed_images)
            loss += self.cfg.loss.tv_loss_weight * tv_loss
        
        # Loss 3: Boundary loss (keep images in [-1, 1])
        if self.cfg.loss.boundary_loss_weight > 0:
            boundary_loss = torch.nn.functional.relu(reconstructed_images.abs() - 1).mean()
            loss += self.cfg.loss.boundary_loss_weight * boundary_loss
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        if self.cfg.optim.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(seer_module.parameters(), self.cfg.optim.grad_clip)
        
        # Optimizer step
        optimizer.step()
        
        # Track statistics
        stats['loss'].append(loss.detach().item())
        
        # Denormalize for output
        reconstructed_images = reconstructed_images.detach()
        reconstructed_images = reconstructed_images * self.ds + self.dm
        reconstructed_images = torch.clamp(reconstructed_images, 0, 1)
        
        return reconstructed_images

    def _compute_gradients(self, model, inputs, labels):
        """Compute per-sample gradients for model on inputs."""
        model.zero_grad()
        
        # Compute per-sample loss
        outputs = model(inputs)
        loss = self.loss_fn(outputs, labels)
        
        # Get per-sample loss (should already be per-sample from loss_fn)
        if loss.dim() > 0:
            batch_loss = loss  # Already per-sample
        else:
            batch_loss = loss.unsqueeze(0)
        
        # Compute gradients
        grads = torch.autograd.grad(
            batch_loss.sum(), 
            model.parameters(), 
            create_graph=True,
            retain_graph=True
        )
        
        return grads

    def _gradient_matching_loss(self, target_grads, reconstructed_grads):
        """Compute MSE between target and reconstructed gradients."""
        loss = 0
        count = 0
        
        for tg, rg in zip(target_grads, reconstructed_grads):
            loss += torch.nn.functional.mse_loss(rg, tg)
            count += 1
        
        return loss / max(count, 1)

    def _total_variation_loss(self, images):
        """Compute total variation regularization for smoothness."""
        if images.shape[-2] < 2 or images.shape[-1] < 2:
            return torch.tensor(0.0, device=images.device, dtype=images.dtype)
        
        diff_x = torch.abs(images[..., 1:, :] - images[..., :-1, :]).mean()
        diff_y = torch.abs(images[..., :, 1:] - images[..., :, :-1]).mean()
        
        return diff_x + diff_y


class SEERModule(nn.Module):
    """Container module combining all SEER components."""
    
    def __init__(self, param_selector, disaggregator, reconstructor):
        super().__init__()
        self.param_selector = param_selector
        self.disaggregator = disaggregator
        self.reconstructor = reconstructor
    
    def forward(self, gradients):
        """Forward pass: gradients -> params_selector -> disaggregator -> reconstructor -> images.
        
        Args:
            gradients: List/Tuple of gradient tensors from model parameters (unbatched or batched)
            
        Returns:
            Flattened reconstructed image data (batch_size, image_pixels)
        """
        # Select sparse parameters directly from gradient list
        sparse_vector = self.param_selector(gradients)
            
        # Pass through disaggregator (encoder)
        hidden_repr = self.disaggregator(sparse_vector)
        
        # Pass through reconstructor (decoder)
        reconstructed = self.reconstructor(hidden_repr)
        
        return reconstructed


class ParamSelector(nn.Module):
    """Selects sparse subset of gradients based on frac and size.
    
    This is crucial for SEER's efficiency - only using 0.1% of parameters.
    """
    
    def __init__(self, model, sz, frac, seed=42, device=torch.device("cpu")):
        super().__init__()
        self.sz = sz
        self.frac = frac
        self.device = device
        
        # Generate random selection for each parameter layer
        try:
            gen = torch.Generator(device=device)
        except RuntimeError:
            gen = torch.Generator() 
            if device.type == 'cuda':
                gen = torch.Generator(device='cuda')

        gen.manual_seed(seed)
        
        total_selected = 0
        param_list = list(model.parameters())
        
        for idx, p in enumerate(param_list):
            num_params = p.numel()
            num_select = max(sz, int(round(num_params * frac)))
            
            # Random permutation and selection
            perm = torch.randperm(num_params, generator=gen, device=device)
            selected_indices = perm[:num_select].sort()[0]
            
            # Register as buffer
            self.register_buffer(f'indices_{idx}', selected_indices)
            total_selected += len(selected_indices)
        
        self.num_par = total_selected
        self.num_layers = len(param_list)
        
        log.info(f"ParamSelector: Selected {self.num_par} params (frac={frac}, size={sz})")

    def forward(self, gradients):
        """Select sparse gradients.
        
        Args:
            gradients: List/Tuple of gradient tensors
            
        Returns:
            Selected sparse gradients (batch_size, selected_params)
        """
        selected_list = []
        
        if isinstance(gradients, (list, tuple)):
            for idx, g in enumerate(gradients):
                indices = getattr(self, f'indices_{idx}')
                
                # Handle batch dimension correctly
                if g.dim() == 1:
                    # 1D tensor (e.g. bias) - add batch dim [1, N]
                    g_flat = g.view(1, -1)
                elif g.dim() > 1:
                    # Check if batched (B, ...) or unbatched (Out, In, ...)
                    # Heuristic: If we are in SEER training, input batch size matters.
                    # Standard FedAvg update is a SINGLE update (batch_size=1 effectively).
                    # So we should treat it as [1, TotalParams].
                    # BUT if we pass a BATCH of gradients (e.g. B updates), then we need B.
                    # For a single image reconstruction, we have ONE update.
                    g_flat = g.flatten().view(1, -1)
                
                # Select indices
                selected = g_flat[:, indices]
                selected_list.append(selected)
        else:
            return gradients # Fallback for single tensor input

        return torch.cat(selected_list, dim=1)


class DeconvDecoder(nn.Module):
    """Deconvolutional decoder for image reconstruction.
    
    Maps from latent hidden representation back to image space.
    """
    
    def __init__(self, input_dim, output_shape, device=torch.device("cpu"), dtype=torch.float32):
        super().__init__()
        self.output_shape = output_shape
        self.device = device
        self.dtype = dtype
        
        # Fully connected layer to initial spatial dimensions
        self.fc = nn.Linear(input_dim, 256 * 4 * 4)
        
        # Deconvolutional layers
        self.deconv1 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.deconv3 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        
        # Final convolution to RGB
        self.conv_final = nn.Conv2d(32, output_shape[0], kernel_size=3, padding=1)
        
        # Move to device
        self.to(device=device, dtype=dtype)
        
        # Move to device
        self.to(device=device, dtype=dtype)

    def forward(self, x):
        """Decode from hidden representation to image.
        
        Args:
            x: Hidden representation (batch_size, input_dim)
            
        Returns:
            Reconstructed images (batch_size, output_size)
        """
        batch_size = x.shape[0]
        
        # FC layer + reshape
        x = self.fc(x)
        x = x.view(batch_size, 256, 4, 4)
        
        # Deconvolutional upsampling
        x = torch.relu(self.deconv1(x))
        x = torch.relu(self.deconv2(x))
        x = torch.relu(self.deconv3(x))
        
        # Final layer: tanh to [-1, 1]
        x = torch.tanh(self.conv_final(x))
        
        # Flatten back to vector
        x = x.view(batch_size, -1)
        
        return x