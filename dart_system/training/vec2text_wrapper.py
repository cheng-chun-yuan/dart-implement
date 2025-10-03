"""
Vec2Text Wrapper for DART Training

Wraps the existing vec2text reconstruction module for training integration
Following Algorithm 1: P_mod ← vec2text(e - n)
"""

import torch
import torch.nn as nn
import logging
from typing import List, Union
from dart_system.reconstruction.vec2text import ChineseVec2Text, ReconstructionConfig

logger = logging.getLogger(__name__)


class Vec2TextWrapper(nn.Module):
    """
    Wrapper for vec2text reconstruction in DART training

    Following Algorithm 1:
    P_mod ← vec2text(e - n)

    where:
    - e: embedding
    - n: noise
    - P_mod: modified/reconstructed prompt
    """

    def __init__(
        self,
        reconstruction_config: ReconstructionConfig = None,
        device: str = "cpu"
    ):
        """
        Initialize vec2text wrapper

        Args:
            reconstruction_config: Configuration for reconstruction
            device: Device for computation
        """
        super().__init__()

        self.device = device
        self.reconstruction_config = reconstruction_config or ReconstructionConfig()

        # Initialize reconstruction model
        self.vec2text_model = ChineseVec2Text(self.reconstruction_config)

        logger.info("Initialized Vec2Text wrapper for DART training")

    def reconstruct(
        self,
        embeddings: torch.Tensor,
        original_prompts: List[str]
    ) -> List[str]:
        """
        Reconstruct prompts from embeddings

        Following Algorithm 1:
        P_mod ← vec2text(e - n)

        where e - n represents the perturbed embedding

        Args:
            embeddings: Perturbed embeddings, shape [batch_size, embedding_dim]
            original_prompts: Original reference prompts for guidance

        Returns:
            List[str]: Reconstructed prompts P_mod
        """
        # Convert torch tensor to list of lists for compatibility
        if isinstance(embeddings, torch.Tensor):
            embeddings_list = embeddings.cpu().detach().tolist()
        else:
            embeddings_list = embeddings

        # Use vec2text decoder to reconstruct
        reconstructed_prompts = self.vec2text_model.decode(
            perturbed_embeddings=embeddings_list,
            original_texts=original_prompts
        )

        return reconstructed_prompts

    def forward(
        self,
        embeddings: torch.Tensor,
        noise: torch.Tensor,
        original_prompts: List[str]
    ) -> List[str]:
        """
        Forward pass: compute P_mod ← vec2text(e - n)

        Args:
            embeddings: Clean embeddings e, shape [batch_size, embedding_dim]
            noise: Sampled noise n, shape [batch_size, embedding_dim]
            original_prompts: Original prompts P

        Returns:
            List[str]: Modified prompts P_mod
        """
        # Compute e - n (perturbed embeddings)
        perturbed_embeddings = embeddings - noise

        # Reconstruct: P_mod ← vec2text(e - n)
        modified_prompts = self.reconstruct(perturbed_embeddings, original_prompts)

        return modified_prompts
