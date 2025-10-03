"""
DART Diffusion Noise Scheduler
Implements the noise scheduling from Algorithm 1

Following Algorithm 1:
n ← N(d_σ(P, e), σ)
where σ is annealed every iteration
"""

import torch
import torch.nn as nn
import math
import logging
from typing import Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class NoiseSchedulerConfig:
    """Configuration for noise scheduler"""
    initial_sigma: float = 1.0
    final_sigma: float = 0.01
    anneal_strategy: str = "linear"  # "linear", "cosine", "exponential"
    device: str = "cpu"


class DiffusionNoiseScheduler:
    """
    Diffusion Noise Scheduler for DART Training

    Implements Algorithm 1 noise generation:
    n ← N(d_σ(P, e), σ)

    where:
    - d_σ: diffusion model
    - P: prompt
    - e: embedding
    - σ: noise level (annealed every iteration)
    """

    def __init__(self, config: NoiseSchedulerConfig):
        """
        Initialize noise scheduler

        Args:
            config: Noise scheduler configuration
        """
        self.config = config
        self.current_sigma = config.initial_sigma
        self.iteration = 0

        logger.info(f"Initialized diffusion noise scheduler")
        logger.info(f"  Initial σ: {config.initial_sigma}")
        logger.info(f"  Final σ: {config.final_sigma}")
        logger.info(f"  Anneal strategy: {config.anneal_strategy}")

    def sample_noise(
        self,
        embeddings: torch.Tensor,
        sigma: Optional[float] = None
    ) -> torch.Tensor:
        """
        Sample noise from diffusion model N(d_σ(P, e), σ)

        Following Algorithm 1:
        n ← N(d_σ(P, e), σ)

        Args:
            embeddings: Input embeddings e, shape [batch_size, embedding_dim]
            sigma: Noise level (uses current_sigma if None)

        Returns:
            torch.Tensor: Sampled noise n, shape [batch_size, embedding_dim]
        """
        if sigma is None:
            sigma = self.current_sigma

        # Get mean from diffusion model d_σ(P, e)
        # For simplicity, we use zero mean (can be extended with learned model)
        mean = torch.zeros_like(embeddings)

        # Sample from N(mean, σ^2 * I)
        noise = torch.randn_like(embeddings) * sigma + mean

        return noise

    def add_noise(
        self,
        embeddings: torch.Tensor,
        noise: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Add noise to embeddings

        Args:
            embeddings: Clean embeddings e
            noise: Pre-sampled noise (samples new if None)

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (noisy embeddings, noise)
        """
        if noise is None:
            noise = self.sample_noise(embeddings)

        noisy_embeddings = embeddings + noise

        return noisy_embeddings, noise

    def anneal_sigma(self, num_epochs: int, current_epoch: int) -> float:
        """
        Anneal sigma following Algorithm 1

        σ is annealed every iteration

        Args:
            num_epochs: Total number of epochs
            current_epoch: Current epoch number

        Returns:
            float: Annealed sigma value
        """
        progress = current_epoch / max(num_epochs - 1, 1)

        if self.config.anneal_strategy == "linear":
            # Linear annealing: σ(t) = σ_0 - (σ_0 - σ_f) * t
            sigma = self.config.initial_sigma - \
                    (self.config.initial_sigma - self.config.final_sigma) * progress

        elif self.config.anneal_strategy == "cosine":
            # Cosine annealing: σ(t) = σ_f + 0.5 * (σ_0 - σ_f) * (1 + cos(π * t))
            sigma = self.config.final_sigma + \
                    0.5 * (self.config.initial_sigma - self.config.final_sigma) * \
                    (1 + math.cos(math.pi * progress))

        elif self.config.anneal_strategy == "exponential":
            # Exponential annealing: σ(t) = σ_0 * (σ_f / σ_0)^t
            sigma = self.config.initial_sigma * \
                    (self.config.final_sigma / self.config.initial_sigma) ** progress

        else:
            raise ValueError(f"Unknown anneal strategy: {self.config.anneal_strategy}")

        self.current_sigma = sigma
        return sigma

    def step(self, num_epochs: int, current_epoch: int):
        """
        Update sigma for current iteration

        Args:
            num_epochs: Total number of epochs
            current_epoch: Current epoch
        """
        self.iteration += 1
        new_sigma = self.anneal_sigma(num_epochs, current_epoch)

        logger.debug(f"Iteration {self.iteration}: σ = {new_sigma:.6f}")

    def get_sigma(self) -> float:
        """Get current sigma value"""
        return self.current_sigma
