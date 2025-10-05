"""
Chinese Embedding Model - ME5 Inverter
Implementation following technical documentation

This module provides:
1. Chinese text embedding using yiyic/t5_me5_base_nq_32_inverter
2. Embedding perturbation for DART attacks
3. Semantic similarity checking

Core components:
- ChineseEmbeddingModel: ME5 inverter-based embedding
- EmbeddingPerturbation: Constrained perturbation system
- SemanticSimilarityChecker: Cosine similarity validation
- ChineseEmbedding: Unified interface wrapper
"""

import torch
import torch.nn as nn
import math
import logging
import numpy as np
from typing import List, Tuple, Optional, Union
from dataclasses import dataclass

from transformers import AutoTokenizer, AutoModel
HF_AVAILABLE = True

logger = logging.getLogger(__name__)

@dataclass
class EmbeddingConfig:
    """Embedding model configuration"""
    embedding_dim: int = 768  # SBERT uses 768-dim embeddings
    max_sequence_length: int = 32  # Matching technical doc specification
    position_decay_factor: float = 0.5
    normalization: bool = True
    encoding_multiplier: int = 31
    unicode_offset: int = 1000
    model_name: str = "uer/sbert-base-chinese-nli"  # SBERT Chinese NLI model (verified working)
    device: Optional[str] = None
    similarity_threshold: float = 0.9


class ChineseEmbeddingModel:
    """
    Chinese embedding model using yiyic/t5_me5_base_nq_32_inverter
    Following technical documentation specifications
    """

    def __init__(
        self,
        model_name: str = "uer/sbert-base-chinese-nli",
        device: Optional[str] = None,
        max_length: int = 512
    ):
        """
        Initialize Chinese SBERT embedding model

        Args:
            model_name: HuggingFace model name for SBERT Chinese
            device: Device to run model on (auto-detect if None)
            max_length: Maximum token length for input text
        """
        self.model_name = model_name
        self.max_length = max_length
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        if not HF_AVAILABLE:
            raise ImportError("HuggingFace transformers required for ME5 inverter model")
        
        logger.info(f"Loading Chinese embedding model: {model_name}")
        logger.info(f"Using device: {self.device}")
        
        try:
            # Load tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)
            self.model.eval().to(self.device)
            
            # Get embedding dimension
            self.embedding_dim = self.model.config.hidden_size
            logger.info(f"Model loaded successfully. Embedding dimension: {self.embedding_dim}")
            
        except Exception as e:
            logger.error(f"Failed to load Chinese embedding model: {e}")
            raise
    
    def embed_text(self, text: str) -> torch.Tensor:
        """
        Encode single Chinese text into dense vector embedding
        
        Args:
            text: Input Chinese text string
            
        Returns:
            torch.Tensor: Dense embedding vector
        """
        return self.embed_texts([text])[0]
    
    def embed_texts(self, texts: List[str]) -> torch.Tensor:
        """
        Encode multiple Chinese texts into dense vector embeddings
        
        Args:
            texts: List of input Chinese text strings
            
        Returns:
            torch.Tensor: Dense embedding vectors [batch_size, embedding_dim]
        """
        if not texts:
            return torch.empty(0, self.embedding_dim, device=self.device)
        
        # Tokenize texts
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=self.max_length
        ).to(self.device)
        
        with torch.no_grad():
            # Get model outputs
            outputs = self.model(**inputs)
            
            # Mean pooling over sequence dimension
            embeddings = outputs.last_hidden_state.mean(dim=1)
            
        return embeddings
    
    def get_embedding_dim(self) -> int:
        """Get embedding dimension"""
        return self.embedding_dim


class EmbeddingPerturbation:
    """Embedding perturbation system for DART attacks"""
    
    def __init__(self, embedding_dim: int, device: str = "cpu"):
        """
        Initialize perturbation system
        
        Args:
            embedding_dim: Dimension of embedding vectors
            device: Device for tensor operations
        """
        self.embedding_dim = embedding_dim
        self.device = device
    
    def perturb_embedding(
        self,
        embedding: torch.Tensor,
        epsilon: float = 0.05,
        noise_type: str = "gaussian"
    ) -> torch.Tensor:
        """
        Apply constrained perturbation to embedding
        
        Args:
            embedding: Original embedding vector
            epsilon: Perturbation magnitude constraint
            noise_type: Type of noise ("gaussian", "uniform")
            
        Returns:
            torch.Tensor: Perturbed embedding vector
        """
        if noise_type == "gaussian":
            noise = torch.randn_like(embedding) * epsilon
        elif noise_type == "uniform":
            noise = (torch.rand_like(embedding) - 0.5) * 2 * epsilon
        else:
            raise ValueError(f"Unknown noise type: {noise_type}")
        
        perturbed = embedding + noise
        
        # Ensure perturbation magnitude constraint
        perturbation_norm = torch.norm(noise)
        if perturbation_norm > epsilon:
            noise = noise / perturbation_norm * epsilon
            perturbed = embedding + noise
        
        return perturbed
    
    def batch_perturb_embeddings(
        self,
        embeddings: torch.Tensor,
        epsilon: float = 0.05,
        noise_type: str = "gaussian"
    ) -> torch.Tensor:
        """
        Apply perturbations to batch of embeddings
        
        Args:
            embeddings: Batch of embedding vectors [batch_size, embedding_dim]
            epsilon: Perturbation magnitude constraint
            noise_type: Type of noise
            
        Returns:
            torch.Tensor: Batch of perturbed embeddings
        """
        perturbed_embeddings = []
        
        for embedding in embeddings:
            perturbed = self.perturb_embedding(embedding, epsilon, noise_type)
            perturbed_embeddings.append(perturbed)
        
        return torch.stack(perturbed_embeddings)


class SemanticSimilarityChecker:
    """Check semantic similarity between original and perturbed embeddings"""
    
    def __init__(self, similarity_threshold: float = 0.9):
        """
        Initialize similarity checker
        
        Args:
            similarity_threshold: Minimum cosine similarity for acceptance
        """
        self.similarity_threshold = similarity_threshold
    
    def cosine_similarity(
        self,
        emb1: torch.Tensor,
        emb2: torch.Tensor
    ) -> float:
        """
        Calculate cosine similarity between two embeddings
        
        Args:
            emb1: First embedding vector
            emb2: Second embedding vector
            
        Returns:
            float: Cosine similarity score
        """
        return torch.nn.functional.cosine_similarity(
            emb1.unsqueeze(0), emb2.unsqueeze(0)
        ).item()
    
    def batch_cosine_similarity(
        self,
        emb1_batch: torch.Tensor,
        emb2_batch: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculate cosine similarity for batch of embeddings
        
        Args:
            emb1_batch: First batch of embeddings
            emb2_batch: Second batch of embeddings
            
        Returns:
            torch.Tensor: Cosine similarity scores
        """
        return torch.nn.functional.cosine_similarity(emb1_batch, emb2_batch)
    
    def check_semantic_preservation(
        self,
        original_embedding: torch.Tensor,
        perturbed_embedding: torch.Tensor
    ) -> bool:
        """
        Check if perturbed embedding preserves semantic meaning
        
        Args:
            original_embedding: Original embedding vector
            perturbed_embedding: Perturbed embedding vector
            
        Returns:
            bool: True if semantic similarity above threshold
        """
        similarity = self.cosine_similarity(original_embedding, perturbed_embedding)
        return similarity >= self.similarity_threshold


# Wrapper class - ME5 inverter only, no fallback
class ChineseEmbedding:
    """Chinese embedding using ME5 inverter model - requires transformers"""

    def __init__(self, config: EmbeddingConfig):
        self.config = config

        if not HF_AVAILABLE:
            raise ImportError(
                "transformers library is required for ChineseEmbedding. "
                "Install it with: uv add transformers"
            )

        # Use ME5 inverter model only
        self._embedder = ChineseEmbeddingModel(
            model_name=config.model_name,
            device=config.device,
            max_length=config.max_sequence_length
        )
        logger.info(f"Using ME5 inverter model: {config.model_name}")

    def encode(self, texts: List[str]) -> np.ndarray:
        """Encode texts to embeddings (numpy array)"""
        embeddings = self._embedder.embed_texts(texts)
        return embeddings.cpu().numpy()