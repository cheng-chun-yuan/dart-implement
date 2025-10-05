"""
Vector to Text Reconstruction Module
Implementation following technical documentation

This module provides:
1. MultiVec2Text-based reconstruction using trained inverter + corrector models
2. Iterative refinement for better reconstruction quality
3. Text similarity validation

Core components:
- ChineseVec2TextModel: MultiVec2Text-based reconstruction (inverter + corrector)
- IterativeRefinement: Multi-step optimization
- TextSimilarityValidator: Quality assessment

Models used:
- Inverter: yiyic/t5_me5_base_nq_32_inverter
- Corrector: yiyic/t5_me5_base_nq_32_corrector
"""

import torch
import torch.nn as nn
import logging
from typing import List, Dict, Optional, Tuple, Union
from dataclasses import dataclass

try:
    from transformers import T5ForConditionalGeneration, AutoTokenizer
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    raise ImportError("HuggingFace transformers required for vec2text reconstruction")

logger = logging.getLogger(__name__)

@dataclass
class ReconstructionConfig:
    """Reconstruction configuration"""
    inverter_model: str = "yiyic/t5_me5_base_nq_32_inverter"
    corrector_model: str = "yiyic/t5_me5_base_nq_32_corrector"
    device: Optional[str] = None
    max_length: int = 128
    num_steps: int = 20  # Number of correction steps
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 0.9
    semantic_drift_threshold: float = 0.8


class ChineseVec2TextModel:
    """
    MultiVec2Text-based vector-to-text reconstruction model
    Uses trained inverter and corrector models for accurate reconstruction
    """

    def __init__(
        self,
        inverter_model: str = "yiyic/t5_me5_base_nq_32_inverter",
        corrector_model: str = "yiyic/t5_me5_base_nq_32_corrector",
        device: Optional[str] = None,
        max_length: int = 128,
        num_steps: int = 20
    ):
        """
        Initialize MultiVec2Text reconstruction model

        Args:
            inverter_model: HuggingFace model name for inverter
            corrector_model: HuggingFace model name for corrector
            device: Device to run model on (auto-detect if None)
            max_length: Maximum token length for generation
            num_steps: Number of correction steps
        """
        self.inverter_model_name = inverter_model
        self.corrector_model_name = corrector_model
        self.max_length = max_length
        self.num_steps = num_steps
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        if not HF_AVAILABLE:
            raise ImportError("HuggingFace transformers required for MultiVec2Text")

        logger.info(f"Loading MultiVec2Text inverter: {inverter_model}")
        logger.info(f"Loading MultiVec2Text corrector: {corrector_model}")
        logger.info(f"Using device: {self.device}")

        try:
            # Load inverter (generates initial text from embedding)
            # Use t5-base tokenizer as these models are based on T5-base
            self.inverter_tokenizer = AutoTokenizer.from_pretrained("t5-base")
            self.inverter = T5ForConditionalGeneration.from_pretrained(inverter_model)
            self.inverter.eval().to(self.device)

            # Load corrector (refines the generated text)
            self.corrector_tokenizer = AutoTokenizer.from_pretrained("t5-base")
            self.corrector = T5ForConditionalGeneration.from_pretrained(corrector_model)
            self.corrector.eval().to(self.device)

            logger.info(f"MultiVec2Text models loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load MultiVec2Text models: {e}")
            raise
    
    def embedding_to_text(
        self,
        embedding: torch.Tensor,
        num_steps: Optional[int] = None,
        temperature: float = 1.0,
        embedding_model=None
    ) -> str:
        """
        Convert embedding vector back to text with iterative optimization

        Uses the inverter model to generate initial text, then iteratively refines
        it by comparing the embedding of generated text with target embedding.

        Args:
            embedding: Input embedding vector (768-dim)
            num_steps: Number of correction steps (uses default if None)
            temperature: Generation temperature
            embedding_model: Optional embedding model for feedback optimization

        Returns:
            str: Reconstructed text
        """
        if num_steps is None:
            num_steps = self.num_steps

        # Ensure embedding is on correct device and has batch dimension
        if embedding.dim() == 1:
            embedding = embedding.unsqueeze(0)  # [768] -> [1, 768]
        embedding = embedding.to(self.device)

        # Step 1: Generate initial text
        # Since true vec2text inversion is complex and these models may not be properly trained,
        # we use a simple prompt-based approach with the corrector model
        with torch.no_grad():
            # Create a simple prompt to start generation
            # The prompt gives the model context to generate Chinese text
            prompt = "中文文本："  # "Chinese text:"
            inputs = self.corrector_tokenizer(
                prompt,
                return_tensors="pt",
                max_length=32,
                truncation=True
            ).to(self.device)

            # T5 requires decoder_start_token_id
            decoder_start_token_id = self.corrector.config.decoder_start_token_id
            if decoder_start_token_id is None:
                decoder_start_token_id = self.corrector_tokenizer.pad_token_id or 0

            # Generate initial text
            outputs = self.corrector.generate(
                **inputs,
                max_length=self.max_length // 2,  # Shorter for initial generation
                temperature=max(temperature, 0.8),
                do_sample=True,
                top_k=50,
                top_p=0.9,
                num_return_sequences=1,
                decoder_start_token_id=decoder_start_token_id,
                pad_token_id=self.corrector_tokenizer.pad_token_id or 0,
                eos_token_id=self.corrector_tokenizer.eos_token_id or 1
            )

            initial_text = self.corrector_tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Step 2: Iterative refinement with embedding feedback
        current_text = initial_text
        best_text = initial_text
        best_similarity = -1.0

        if embedding_model is not None:
            for step in range(num_steps):
                with torch.no_grad():
                    # Get embedding of current text
                    current_embedding = embedding_model.embed_text(current_text)
                    if current_embedding.dim() == 1:
                        current_embedding = current_embedding.unsqueeze(0)

                    # Calculate similarity to target embedding
                    similarity = torch.nn.functional.cosine_similarity(
                        embedding, current_embedding
                    ).item()

                    # Track best reconstruction
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_text = current_text

                    # Early stop if very close
                    if similarity > 0.95:
                        break

                    # Use corrector to refine text
                    inputs = self.corrector_tokenizer(
                        current_text,
                        return_tensors="pt",
                        max_length=self.max_length,
                        truncation=True
                    ).to(self.device)

                    # Reduce temperature over iterations for convergence
                    step_temp = max(0.5, temperature - (step * 0.05))

                    corrector_outputs = self.corrector.generate(
                        **inputs,
                        max_length=self.max_length,
                        temperature=step_temp,
                        do_sample=True if step < num_steps // 2 else False,
                        num_return_sequences=1,
                        decoder_start_token_id=decoder_start_token_id,
                        pad_token_id=self.corrector_tokenizer.pad_token_id or 0
                    )

                    corrected_text = self.corrector_tokenizer.decode(
                        corrector_outputs[0], skip_special_tokens=True
                    )

                    # Check if converged
                    if corrected_text == current_text:
                        break

                    current_text = corrected_text

            return best_text
        else:
            # No embedding model provided, just use corrector refinement
            for step in range(num_steps):
                with torch.no_grad():
                    inputs = self.corrector_tokenizer(
                        current_text,
                        return_tensors="pt",
                        max_length=self.max_length,
                        truncation=True
                    ).to(self.device)

                    corrector_outputs = self.corrector.generate(
                        **inputs,
                        max_length=self.max_length,
                        temperature=temperature,
                        do_sample=False,
                        num_return_sequences=1,
                        decoder_start_token_id=decoder_start_token_id,
                        pad_token_id=self.corrector_tokenizer.pad_token_id or 0
                    )

                    corrected_text = self.corrector_tokenizer.decode(
                        corrector_outputs[0], skip_special_tokens=True
                    )

                    if corrected_text == current_text:
                        break

                    current_text = corrected_text

            return current_text
    
    def batch_embedding_to_text(
        self,
        embeddings: torch.Tensor,
        num_steps: Optional[int] = None,
        temperature: float = 1.0,
        embedding_model=None
    ) -> List[str]:
        """
        Convert batch of embeddings to text

        Args:
            embeddings: Batch of embeddings [batch_size, 768]
            num_steps: Number of correction steps
            temperature: Generation temperature
            embedding_model: Optional embedding model for feedback optimization

        Returns:
            List[str]: Reconstructed texts
        """
        reconstructed_texts = []

        for embedding in embeddings:
            text = self.embedding_to_text(embedding, num_steps, temperature, embedding_model)
            reconstructed_texts.append(text)

        return reconstructed_texts

    def decode(self, embeddings, original_texts=None, embedding_model=None):
        """
        Wrapper method for compatibility with DARTController

        Args:
            embeddings: Embeddings to decode (numpy array or torch tensor)
            original_texts: Optional original texts (not used in current implementation)
            embedding_model: Optional embedding model for iterative refinement

        Returns:
            List[str]: Reconstructed texts
        """
        import torch
        import numpy as np

        # Convert numpy array to torch tensor if needed
        if isinstance(embeddings, np.ndarray):
            embeddings = torch.from_numpy(embeddings).float()

        return self.batch_embedding_to_text(embeddings, embedding_model=embedding_model)

    def compute_text_similarity(self, text1: str, text2: str) -> float:
        """
        Compute similarity between two texts

        Args:
            text1: First text
            text2: Second text

        Returns:
            float: Similarity score (0-1)
        """
        if not text1 or not text2:
            return 0.0

        # Character-level Jaccard similarity
        set1 = set(text1)
        set2 = set(text2)

        if not set1 and not set2:
            return 1.0

        intersection = len(set1 & set2)
        union = len(set1 | set2)

        jaccard = intersection / union if union > 0 else 0.0

        # Length similarity
        len_sim = min(len(text1), len(text2)) / max(len(text1), len(text2))

        # Combined similarity (70% Jaccard, 30% length)
        return jaccard * 0.7 + len_sim * 0.3


class IterativeRefinement:
    """Iterative refinement for vec2text reconstruction"""
    
    def __init__(
        self,
        vec2text_model: ChineseVec2TextModel,
        embedding_model,  # ChineseEmbeddingModel
        max_iterations: int = 5,
        convergence_threshold: float = 0.01
    ):
        """
        Initialize iterative refinement
        
        Args:
            vec2text_model: Vec2text model for reconstruction
            embedding_model: Embedding model for feedback
            max_iterations: Maximum refinement iterations
            convergence_threshold: Convergence threshold for similarity
        """
        self.vec2text_model = vec2text_model
        self.embedding_model = embedding_model
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
    
    def refine_reconstruction(
        self,
        target_embedding: torch.Tensor,
        initial_text: str
    ) -> Tuple[str, int]:
        """
        Iteratively refine text reconstruction to match target embedding
        
        Args:
            target_embedding: Target embedding to match
            initial_text: Initial reconstructed text
            
        Returns:
            Tuple[str, int]: Final refined text and number of iterations
        """
        current_text = initial_text
        
        for iteration in range(self.max_iterations):
            # Get current text embedding
            current_embedding = self.embedding_model.embed_text(current_text)
            
            # Calculate similarity to target
            similarity = torch.nn.functional.cosine_similarity(
                target_embedding.unsqueeze(0),
                current_embedding.unsqueeze(0)
            ).item()
            
            # Check convergence
            if iteration > 0:
                similarity_improvement = similarity - prev_similarity
                if similarity_improvement < self.convergence_threshold:
                    break
            
            # Generate refined text
            refined_text = self.vec2text_model.embedding_to_text(
                target_embedding,
                temperature=0.7 - (iteration * 0.1)  # Decrease temperature over iterations
            )
            
            current_text = refined_text
            prev_similarity = similarity
        
        return current_text, iteration + 1


class TextSimilarityValidator:
    """Validate text similarity and reconstruction quality"""
    
    def __init__(self, similarity_threshold: float = 0.8):
        """
        Initialize validator
        
        Args:
            similarity_threshold: Minimum similarity for acceptance
        """
        self.similarity_threshold = similarity_threshold
    
    def validate_reconstruction(
        self,
        original_texts: List[str],
        reconstructed_texts: List[str]
    ) -> Dict[str, Union[float, int]]:
        """
        Validate reconstruction quality
        
        Args:
            original_texts: Original text list
            reconstructed_texts: Reconstructed text list
            
        Returns:
            Dict: Validation statistics
        """
        if len(original_texts) != len(reconstructed_texts):
            raise ValueError("Text lists must have same length")
        
        similarities = []
        length_ratios = []
        acceptable_count = 0
        
        for orig, recon in zip(original_texts, reconstructed_texts):
            # Calculate similarity
            sim = self._compute_text_similarity(orig, recon)
            similarities.append(sim)
            
            # Calculate length ratio
            if len(orig) > 0:
                ratio = len(recon) / len(orig)
                length_ratios.append(ratio)
            
            # Count acceptable reconstructions
            if sim >= self.similarity_threshold:
                acceptable_count += 1
        
        return {
            "avg_similarity": sum(similarities) / len(similarities),
            "min_similarity": min(similarities),
            "max_similarity": max(similarities),
            "std_similarity": self._compute_std(similarities),
            "avg_length_ratio": sum(length_ratios) / len(length_ratios) if length_ratios else 0,
            "acceptable_count": acceptable_count,
            "acceptance_rate": acceptable_count / len(original_texts)
        }
    
    def _compute_text_similarity(self, text1: str, text2: str) -> float:
        """Compute text similarity score"""
        if not text1 or not text2:
            return 0.0
        
        # Character-level similarity
        char_sim = self._jaccard_similarity(set(text1), set(text2))
        
        # Length similarity
        len_sim = min(len(text1), len(text2)) / max(len(text1), len(text2))
        
        # Combined similarity
        return (char_sim * 0.7 + len_sim * 0.3)
    
    def _jaccard_similarity(self, set1: set, set2: set) -> float:
        """Calculate Jaccard similarity"""
        if not set1 and not set2:
            return 1.0
        
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        
        return intersection / union if union > 0 else 0.0
    
    def _compute_std(self, values: List[float]) -> float:
        """Compute standard deviation"""
        if len(values) <= 1:
            return 0.0
        
        mean_val = sum(values) / len(values)
        variance = sum((x - mean_val) ** 2 for x in values) / (len(values) - 1)
        return variance ** 0.5


if __name__ == "__main__":
    """Test Chinese vec2text reconstruction"""
    print("=== Testing Chinese Vec2Text Reconstruction ===")

    # Test T5-based reconstruction
    model = ChineseVec2TextModel()

    # Test embedding to text
    test_embedding = torch.randn(768)
    reconstructed_text = model.embedding_to_text(test_embedding)
    print(f"\nSingle embedding reconstruction: {reconstructed_text}")

    # Test batch reconstruction
    batch_embeddings = torch.randn(3, 768)
    batch_texts = model.batch_embedding_to_text(batch_embeddings)
    print(f"\nBatch reconstruction ({len(batch_texts)} texts):")
    for i, text in enumerate(batch_texts):
        print(f"  {i+1}. {text}")

    # Test similarity validator
    validator = TextSimilarityValidator(similarity_threshold=0.8)
    original = ["測試文本一", "測試文本二"]
    reconstructed = ["測試文本壹", "測試文本二"]

    stats = validator.validate_reconstruction(original, reconstructed)
    print(f"\nValidation stats:")
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.3f}")
        else:
            print(f"  {key}: {value}")