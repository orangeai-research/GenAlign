"""
Rectified Flow Pluggable Module

This module provides a standalone, reusable implementation of Rectified Flow (RF)
that can be easily integrated into any recommendation network to enhance embeddings.

Key Components:
- SinusoidalPositionEmbedding: Time step encoding
- ResidualBlock: Residual block for network depth
- SimpleVelocityNet: Velocity field network for RF
- cosine_similarity_gradient: Cosine similarity gradient computation
- RFEmbeddingGenerator: Main pluggable RF generator class

Usage Example:
    ```python
    from models.rf_modules import RFEmbeddingGenerator

    # Initialize RF generator
    rf_gen = RFEmbeddingGenerator(
        embedding_dim=64,
        hidden_dim=256,
        n_layers=2,
        learning_rate=0.001,
    )

    # Training: compute loss and update
    loss_dict = rf_gen.compute_loss_and_step(
        target_embeds=target,
        conditions=[image_cond, text_cond],
        user_prior=prior,
        epoch=current_epoch,
    )

    # Generate embeddings
    rf_embeds = rf_gen.generate([image_cond, text_cond])

    # Mix with original embeddings
    final_embeds = rf_gen.mix_embeddings(
        original_embeds, rf_embeds, training=True
    )
    ```
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Dict


def cosine_similarity_gradient(x_t: torch.Tensor, x_1: torch.Tensor) -> torch.Tensor:
    """
    Compute the gradient of cosine similarity cos_sim(x_t, x_1) with respect to x_t.

    Gradient formula: ∇_{x_t} cos_sim(x_t, x_1) = (x_1 / ||x_1||) / ||x_t|| - (x_t / ||x_t||) * cos_sim / ||x_t||

    Args:
        x_t: Current state, shape (batch, embedding_dim)
        x_1: Target state, shape (batch, embedding_dim)

    Returns:
        grad: Cosine similarity gradient, shape (batch, embedding_dim)
    """
    # Compute cosine similarity
    cos_sim = F.cosine_similarity(x_t, x_1, dim=-1)  # (batch,)
    cos_sim = cos_sim.unsqueeze(-1)  # (batch, 1)

    # Compute norms
    x_t_norm = x_t.norm(dim=-1, keepdim=True)  # (batch, 1)
    x_t_norm = torch.clamp(x_t_norm, min=1e-8)  # Avoid division by zero

    # Normalized vectors
    x_1_normalized = F.normalize(x_1, dim=-1)  # (batch, embedding_dim)
    x_t_normalized = F.normalize(x_t, dim=-1)  # (batch, embedding_dim)

    # Compute gradient
    grad = x_1_normalized / x_t_norm - x_t_normalized * cos_sim / x_t_norm

    return grad


class SinusoidalPositionEmbedding(nn.Module):
    """Sinusoidal position encoding for time steps."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: Time step, shape (batch, 1)
        Returns:
            embeddings: shape (batch, dim)
        """
        device = t.device
        half_dim = self.dim // 2

        # 确保所有 tensor 都在同一设备上
        embeddings = torch.log(torch.tensor(10000.0, device=device)) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = t * embeddings[None, :]

        embeddings = torch.cat([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)

        return embeddings


class ResidualBlock(nn.Module):
    """Simple residual block with LayerNorm and SiLU activation."""

    def __init__(self, in_dim: int, out_dim: int, dropout: float):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim, out_dim),
            nn.LayerNorm(out_dim),
        )

        self.skip = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.silu(self.net(x) + self.skip(x))


class SimpleVelocityNet(nn.Module):
    """
    Simple velocity field network for Rectified Flow.

    Args:
        embedding_dim: Embedding dimension
        hidden_dim: Hidden layer dimension
        n_layers: Number of layers
        dropout: Dropout rate
        condition_dim: Condition dimension (sum of all condition embeddings)
        user_guidance_scale: User prior guidance scaling factor
        guidance_decay_power: User prior decay power
        cosine_guidance_scale: Cosine gradient guidance scaling factor
        cosine_decay_power: Cosine gradient decay power
    """

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        n_layers: int,
        dropout: float,
        condition_dim: int,
        user_guidance_scale: float = 0.2,
        guidance_decay_power: float = 2.0,
        cosine_guidance_scale: float = 0.1,
        cosine_decay_power: float = 2.0,
    ):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.user_guidance_scale = user_guidance_scale
        self.guidance_decay_power = guidance_decay_power
        self.cosine_guidance_scale = cosine_guidance_scale
        self.cosine_decay_power = cosine_decay_power

        # Time embedding layer (using sinusoidal position encoding)
        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbedding(64),
            nn.Linear(64, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
        )

        # Condition encoder
        self.condition_encoder = nn.Sequential(
            nn.Linear(condition_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
        )

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
        )

        # Residual blocks
        self.res_blocks = nn.ModuleList(
            [ResidualBlock(hidden_dim, hidden_dim, dropout) for _ in range(n_layers)]
        )

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embedding_dim),
        )

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        conditions: torch.Tensor,
        user_prior: Optional[torch.Tensor] = None,
        x_1: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: Current state X_t, shape (batch, embedding_dim)
            t: Time step, shape (batch, 1)
            conditions: Condition features (concatenated), shape (batch, condition_dim)
            user_prior: User-specific interest guidance, shape (batch, embedding_dim)
            x_1: Target state X_1, shape (batch, embedding_dim) - for cosine gradient

        Returns:
            velocity: Velocity v(x, t, c), shape (batch, embedding_dim)
        """
        # Time embedding
        t_emb = self.time_embed(t)  # (batch, hidden_dim)

        # Condition embedding
        cond_emb = self.condition_encoder(conditions)  # (batch, hidden_dim)

        # Input projection
        h = self.input_proj(x)  # (batch, hidden_dim)

        # Fuse time and condition
        h = h + t_emb + cond_emb

        # Pass through residual blocks
        for res_block in self.res_blocks:
            h = res_block(h)

        # Output velocity
        v = self.output_proj(h)

        # Add prior knowledge guidance (training mode only)
        if self.training:
            # First term: User interest prior guidance
            if user_prior is not None:
                # Time decay weight: lambda_1(t) = (1-t)^power
                lambda_1 = (1 - t) ** self.guidance_decay_power  # shape: (batch, 1)

                # Add user prior guidance term
                v = v + lambda_1 * self.user_guidance_scale * user_prior

            # Second term: Cosine similarity gradient guidance
            if x_1 is not None:
                # Time decay weight: lambda_2(t) = (1-t)^power
                lambda_2 = (1 - t) ** self.cosine_decay_power  # shape: (batch, 1)

                # Compute cosine similarity gradient
                cos_grad = cosine_similarity_gradient(x, x_1)

                # Add cosine gradient guidance term
                v = v + lambda_2 * self.cosine_guidance_scale * cos_grad

        return v


class RFEmbeddingGenerator(nn.Module):
    """
    Pluggable Rectified Flow Embedding Generator.

    This is the main class for integrating RF into recommendation networks.
    It provides a clean API for training and inference.

    Args:
        embedding_dim: Embedding dimension
        hidden_dim: Hidden layer dimension (default: 256)
        n_layers: Number of network layers (default: 2)
        dropout: Dropout rate (default: 0.1)
        learning_rate: Independent learning rate for RF (default: 0.001)
        sampling_steps: ODE sampling steps (default: 10)
        user_guidance_scale: User prior scaling factor (default: 0.2)
        guidance_decay_power: User prior decay power (default: 2.0)
        cosine_guidance_scale: Cosine gradient scaling factor (default: 0.1)
        cosine_decay_power: Cosine gradient decay power (default: 2.0)
        warmup_epochs: Warmup epochs (default: 0)
        train_mix_ratio: Training mix ratio (default: 0.5)
        inference_mix_ratio: Inference mix ratio (default: 0.5)
        contrast_temp: Contrastive loss temperature (default: 0.2)
        contrast_weight: Contrastive loss weight (default: 1.0)
    """

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int = 256,
        n_layers: int = 2,
        dropout: float = 0.1,
        learning_rate: float = 0.001,
        sampling_steps: int = 10,
        user_guidance_scale: float = 0.2,
        guidance_decay_power: float = 2.0,
        cosine_guidance_scale: float = 0.1,
        cosine_decay_power: float = 2.0,
        warmup_epochs: int = 0,
        train_mix_ratio: float = 0.5,
        inference_mix_ratio: float = 0.5,
        contrast_temp: float = 0.2,
        contrast_weight: float = 1.0,
    ):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.sampling_steps = sampling_steps
        self.user_guidance_scale = user_guidance_scale
        self.guidance_decay_power = guidance_decay_power
        self.cosine_guidance_scale = cosine_guidance_scale
        self.cosine_decay_power = cosine_decay_power
        self.warmup_epochs = warmup_epochs
        self.train_mix_ratio = train_mix_ratio
        self.inference_mix_ratio = inference_mix_ratio
        self.contrast_temp = contrast_temp
        self.contrast_weight = contrast_weight

        # Will be set when we know the condition dimension
        self.velocity_net = None
        self.optimizer = None

        # Current epoch (updated externally)
        self.current_epoch = 0

    def _init_velocity_net(self, condition_dim: int, device: torch.device):
        """Initialize velocity network (called on first forward pass)."""
        if self.velocity_net is None:
            self.velocity_net = SimpleVelocityNet(
                embedding_dim=self.embedding_dim,
                hidden_dim=self.hidden_dim,
                n_layers=self.n_layers,
                dropout=self.dropout,
                condition_dim=condition_dim,
                user_guidance_scale=self.user_guidance_scale,
                guidance_decay_power=self.guidance_decay_power,
                cosine_guidance_scale=self.cosine_guidance_scale,
                cosine_decay_power=self.cosine_decay_power,
            )

            # Move velocity_net to the correct device
            self.velocity_net = self.velocity_net.to(device)

            # Initialize optimizer
            self.optimizer = torch.optim.AdamW(
                self.velocity_net.parameters(),
                lr=self.learning_rate
            )

    def _rectified_flow_loss(
        self,
        target_embeds: torch.Tensor,
        conditions: torch.Tensor,
        user_prior: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute Rectified Flow loss with optional user interest prior guidance.

        Loss = E_t [||v(X_t, t, c) - (X1 - X0)||^2]
        where X_t = t*X1 + (1-t)*X0

        Args:
            target_embeds: Target embeddings (X1), shape (batch, embedding_dim)
            conditions: Concatenated condition embeddings, shape (batch, condition_dim)
            user_prior: User interest prior guidance, shape (batch, embedding_dim)

        Returns:
            rf_loss: Rectified Flow loss
        """
        batch_size = target_embeds.shape[0]

        # X1 = target embeddings
        X1 = target_embeds

        # X0 = random noise
        X0 = torch.randn_like(X1)

        # Sample time step t ~ Uniform[0, 1]
        t = torch.rand(batch_size, 1).to(X1.device)

        # Linear interpolation: X_t = t*X1 + (1-t)*X0
        X_t = t * X1 + (1 - t) * X0

        # Predict velocity: v(X_t, t, conditions, user_prior, X1)
        v_pred = self.velocity_net(X_t, t, conditions, user_prior=user_prior, x_1=X1)

        # Target velocity: X1 - X0 (straight line direction)
        v_target = X1 - X0

        # Rectified Flow loss (MSE)
        rf_loss = F.mse_loss(v_pred, v_target)

        return rf_loss

    def _infonce_loss(
        self,
        view1: torch.Tensor,
        view2: torch.Tensor,
        temperature: float,
    ) -> torch.Tensor:
        """
        Compute InfoNCE contrastive loss (matching mgcn.py implementation).

        Args:
            view1: First view embeddings
            view2: Second view embeddings
            temperature: Temperature parameter

        Returns:
            loss: InfoNCE loss
        """
        # Normalize embeddings
        view1, view2 = F.normalize(view1, dim=1), F.normalize(view2, dim=1)

        # Positive score: element-wise product and sum
        pos_score = (view1 * view2).sum(dim=-1)
        pos_score = torch.exp(pos_score / temperature)

        # Total score: matrix multiplication
        ttl_score = torch.matmul(view1, view2.transpose(0, 1))
        ttl_score = torch.exp(ttl_score / temperature).sum(dim=1)

        # InfoNCE loss
        cl_loss = -torch.log(pos_score / ttl_score)

        return torch.mean(cl_loss)

    def compute_loss_and_step(
        self,
        target_embeds: torch.Tensor,
        conditions: List[torch.Tensor],
        user_prior: Optional[torch.Tensor] = None,
        epoch: Optional[int] = None,
    ) -> Dict[str, float]:
        """
        Compute RF velocity field loss and execute optimization step.

        Note: Contrastive loss (cl_loss) is now computed in the main model's
        calculate_loss method, not here.

        Args:
            target_embeds: Target embedding (RF learning target), shape (batch, embedding_dim)
            conditions: List of condition embeddings (will be concatenated)
            user_prior: Optional user prior guidance, shape (batch, embedding_dim)
            epoch: Current epoch (for warmup control)

        Returns:
            loss_dict: {"rf_loss": float}
        """
        # Update epoch if provided
        if epoch is not None:
            self.current_epoch = epoch

        # Concatenate conditions
        conditions_cat = torch.cat(conditions, dim=-1)

        # Initialize velocity network if needed
        self._init_velocity_net(conditions_cat.shape[-1], target_embeds.device)

        # Ensure velocity_net is in training mode
        self.velocity_net.train()

        # Compute RF velocity field loss
        rf_loss = self._rectified_flow_loss(
            target_embeds.detach(),
            conditions_cat.detach(),
            user_prior=user_prior.detach() if user_prior is not None else None,
        )

        # RF independent backpropagation and parameter update
        self.optimizer.zero_grad()
        rf_loss.backward()
        self.optimizer.step()

        return {"rf_loss": rf_loss.item()}

    def generate(
        self,
        conditions: List[torch.Tensor],
        n_steps: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Generate embeddings using ODE sampling.

        Args:
            conditions: List of condition embeddings (will be concatenated)
            n_steps: ODE sampling steps (None to use default value)

        Returns:
            generated_embeds: Generated embeddings
        """
        # Concatenate conditions
        conditions_cat = torch.cat(conditions, dim=-1)

        # Get batch size and device
        batch_size = conditions_cat.shape[0]
        device = conditions_cat.device

        # Initialize velocity network if needed
        self._init_velocity_net(conditions_cat.shape[-1], device)

        # Use default sampling steps if not provided
        if n_steps is None:
            n_steps = self.sampling_steps

        # Start from standard Gaussian noise
        z_0 = torch.randn(batch_size, self.embedding_dim).to(device)

        # Solve ODE using Euler method
        z_t = z_0
        dt = 1.0 / n_steps

        # Set to eval mode for inference
        is_training = self.velocity_net.training
        self.velocity_net.eval()

        with torch.set_grad_enabled(is_training):
            for i in range(n_steps):
                t = torch.full((batch_size, 1), i * dt).to(device)
                v = self.velocity_net(z_t, t, conditions_cat)
                z_t = z_t + v * dt

        # Restore training mode
        if is_training:
            self.velocity_net.train()

        return z_t

    def mix_embeddings(
        self,
        original_embeds: torch.Tensor,
        generated_embeds: torch.Tensor,
        training: bool = True,
        epoch: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Mix original and generated embeddings.

        Automatically decides mixing strategy based on warmup status and mix_ratio.

        Args:
            original_embeds: Original embedding
            generated_embeds: RF-generated embedding
            training: Training/inference mode
            epoch: Current epoch (for warmup control)

        Returns:
            mixed_embeds: Mixed embeddings
        """
        # Update epoch if provided
        if epoch is not None:
            self.current_epoch = epoch

        # During warmup: use pure original embeddings
        if self.current_epoch < self.warmup_epochs:
            return original_embeds

        # After warmup: mix based on ratio
        if training:
            mix_ratio = self.train_mix_ratio
        else:
            mix_ratio = self.inference_mix_ratio

        mixed_embeds = (1 - mix_ratio) * original_embeds + mix_ratio * generated_embeds

        return mixed_embeds

    def set_epoch(self, epoch: int):
        """Set current epoch (called by trainer)."""
        self.current_epoch = epoch
