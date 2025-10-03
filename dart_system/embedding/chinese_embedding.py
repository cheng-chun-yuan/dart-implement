"""
Chinese Embedding Model - SBERT-based Chinese NLI
Implementation following technical documentation

This module provides:
1. Chinese text embedding using uer/sbert-base-chinese-nli
2. Embedding perturbation for DART attacks
3. Semantic similarity checking
4. Support for both HuggingFace models and fallback Unicode-based encoding

Core components:
- ChineseEmbeddingModel: SBERT-based embedding
- EmbeddingPerturbation: Constrained perturbation system
- SemanticSimilarityChecker: Cosine similarity validation
- FallbackChineseEmbedding: Unicode-based backup implementation
"""

import torch
import torch.nn as nn
import math
import logging
import numpy as np
from typing import List, Tuple, Optional, Union
from dataclasses import dataclass

try:
    from transformers import AutoTokenizer, AutoModel
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    logging.warning("HuggingFace transformers not available, using fallback implementation")

logger = logging.getLogger(__name__)

@dataclass
class EmbeddingConfig:
    """Embedding model configuration"""
    embedding_dim: int = 512
    max_sequence_length: int = 32  # Following technical doc
    position_decay_factor: float = 0.5
    normalization: bool = True
    encoding_multiplier: int = 31
    unicode_offset: int = 1000
    model_name: str = "uer/sbert-base-chinese-nli"
    device: Optional[str] = None
    similarity_threshold: float = 0.9


class ChineseEmbeddingModel:
    """
    Chinese embedding model using uer/sbert-base-chinese-nli
    Following technical documentation specifications
    """
    
    def __init__(
        self,
        model_name: str = "uer/sbert-base-chinese-nli",
        device: Optional[str] = None,
        max_length: int = 32
    ):
        """
        Initialize Chinese SBERT embedding model
        
        Args:
            model_name: HuggingFace model name for Chinese SBERT
            device: Device to run model on (auto-detect if None)
            max_length: Maximum token length for input text
        """
        self.model_name = model_name
        self.max_length = max_length
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        if not HF_AVAILABLE:
            raise ImportError("HuggingFace transformers required for SBERT model")
        
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


# Fallback implementation (rename existing class)
class FallbackChineseEmbedding:
    """
    中文嵌入編碼器
    
    將中文文本轉換為固定維度的向量表示，
    考慮字符的Unicode編碼和在文本中的位置。
    """
    
    def __init__(self, config: Optional[EmbeddingConfig] = None):
        """
        初始化中文嵌入器
        
        Args:
            config: 嵌入配置，如果為None則使用默認配置
        """
        self.config = config or EmbeddingConfig()
        self.embedding_dim = self.config.embedding_dim
        self.max_length = self.config.max_sequence_length
        
        logger.info(f"Initialized ChineseEmbedding with dim={self.embedding_dim}")
    
    def encode(self, texts: List[str]) -> np.ndarray:
        """
        將文本列表編碼為嵌入向量列表

        Args:
            texts: 中文文本列表

        Returns:
            np.ndarray: 嵌入向量數組 [batch_size, embedding_dim]
        """
        logger.debug(f"Encoding {len(texts)} texts to embeddings")

        embeddings = []
        for text in texts:
            embedding = self._text_to_embedding(text)
            embeddings.append(embedding)

        return np.array(embeddings, dtype=np.float32)
    
    def encode_single(self, text: str) -> List[float]:
        """
        編碼單個文本
        
        Args:
            text: 中文文本
            
        Returns:
            List[float]: 嵌入向量
        """
        return self._text_to_embedding(text)
    
    def _text_to_embedding(self, text: str) -> List[float]:
        """
        將單個中文文本轉換為嵌入向量
        
        核心算法:
        1. 遍歷文本中的每個字符
        2. 計算字符的Unicode編碼特徵
        3. 應用位置權重衰減
        4. 累加到向量的每個維度
        5. L2正規化
        
        Args:
            text: 中文文本
            
        Returns:
            List[float]: 正規化後的嵌入向量
        """
        # 截斷過長的文本
        chars = list(text)[:self.max_length]
        
        # 初始化零向量
        embedding = [0.0] * self.embedding_dim
        
        # 遍歷每個字符
        for char_idx, char in enumerate(chars):
            # 獲取字符的Unicode編碼
            char_code = ord(char)
            
            # 計算位置權重（位置越靠後權重越小）
            position_weight = self._calculate_position_weight(char_idx)
            
            # 為每個維度計算特徵值
            for dim_idx in range(self.embedding_dim):
                # 基於字符編碼和維度索引生成特徵
                feature_value = self._calculate_feature(char_code, dim_idx)
                # 累加加權特徵到對應維度
                embedding[dim_idx] += feature_value * position_weight
        
        # L2正規化
        if self.config.normalization:
            embedding = self._normalize_vector(embedding)
        
        return embedding
    
    def _calculate_position_weight(self, position: int) -> float:
        """
        計算位置權重
        
        使用平方根衰減: 1 / sqrt(position + 1)
        
        Args:
            position: 字符在文本中的位置（從0開始）
            
        Returns:
            float: 位置權重值
        """
        return 1.0 / ((position + 1) ** self.config.position_decay_factor)
    
    def _calculate_feature(self, char_code: int, dim_idx: int) -> float:
        """
        計算字符在特定維度的特徵值
        
        算法: ((char_code + dim_idx * multiplier) % offset) / offset
        
        Args:
            char_code: 字符的Unicode編碼
            dim_idx: 維度索引
            
        Returns:
            float: 特徵值 (0-1之間)
        """
        multiplier = self.config.encoding_multiplier
        offset = self.config.unicode_offset
        
        # 計算特徵值，確保在0-1範圍內
        feature = ((char_code + dim_idx * multiplier) % offset) / offset
        return feature
    
    def _normalize_vector(self, vector: List[float]) -> List[float]:
        """
        L2正規化向量
        
        將向量標準化為單位向量: v_norm = v / ||v||_2
        
        Args:
            vector: 原始向量
            
        Returns:
            List[float]: 正規化後的向量
        """
        # 計算L2範數
        l2_norm = math.sqrt(sum(x * x for x in vector))
        
        # 避免除零錯誤
        if l2_norm == 0.0:
            return vector
        
        # 正規化
        return [x / l2_norm for x in vector]
    
    def compute_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """
        計算兩個向量的餘弦相似度
        
        公式: cos(θ) = (v1 · v2) / (||v1|| * ||v2||)
        
        Args:
            vec1: 第一個向量
            vec2: 第二個向量
            
        Returns:
            float: 餘弦相似度值 (-1 到 1)
        """
        if len(vec1) != len(vec2):
            raise ValueError(f"Vector dimensions mismatch: {len(vec1)} vs {len(vec2)}")
        
        # 計算點積
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        
        # 計算向量的模長
        norm1 = math.sqrt(sum(x * x for x in vec1))
        norm2 = math.sqrt(sum(x * x for x in vec2))
        
        # 避免除零錯誤
        if norm1 == 0.0 or norm2 == 0.0:
            return 0.0
        
        # 餘弦相似度
        similarity = dot_product / (norm1 * norm2)
        return similarity
    
    def compute_l2_distance(self, vec1: List[float], vec2: List[float]) -> float:
        """
        計算兩個向量的L2距離
        
        公式: ||v1 - v2||_2 = sqrt(sum((v1_i - v2_i)^2))
        
        Args:
            vec1: 第一個向量
            vec2: 第二個向量
            
        Returns:
            float: L2距離
        """
        if len(vec1) != len(vec2):
            raise ValueError(f"Vector dimensions mismatch: {len(vec1)} vs {len(vec2)}")
        
        # 計算平方差的和
        squared_diff_sum = sum((a - b) ** 2 for a, b in zip(vec1, vec2))
        
        # 返回平方根
        return math.sqrt(squared_diff_sum)
    
    def batch_similarity(self, embeddings: List[List[float]]) -> List[List[float]]:
        """
        計算一批嵌入向量之間的相似度矩陣
        
        Args:
            embeddings: 嵌入向量列表
            
        Returns:
            List[List[float]]: 相似度矩陣
        """
        n = len(embeddings)
        similarity_matrix = []
        
        for i in range(n):
            row = []
            for j in range(n):
                if i == j:
                    row.append(1.0)  # 自己與自己的相似度為1
                else:
                    sim = self.compute_similarity(embeddings[i], embeddings[j])
                    row.append(sim)
            similarity_matrix.append(row)
        
        return similarity_matrix
    
    def get_embedding_statistics(self, embeddings: List[List[float]]) -> dict:
        """
        計算嵌入向量的統計信息
        
        Args:
            embeddings: 嵌入向量列表
            
        Returns:
            dict: 統計信息
        """
        if not embeddings:
            return {}
        
        # 計算每個維度的統計信息
        dim_means = []
        dim_stds = []
        
        for dim in range(len(embeddings[0])):
            values = [emb[dim] for emb in embeddings]
            mean_val = sum(values) / len(values)
            variance = sum((x - mean_val) ** 2 for x in values) / len(values)
            std_val = math.sqrt(variance)
            
            dim_means.append(mean_val)
            dim_stds.append(std_val)
        
        # 計算向量長度統計
        norms = [math.sqrt(sum(x * x for x in emb)) for emb in embeddings]
        
        return {
            "num_vectors": len(embeddings),
            "vector_dim": len(embeddings[0]),
            "mean_norm": sum(norms) / len(norms),
            "std_norm": math.sqrt(sum((x - sum(norms)/len(norms)) ** 2 for x in norms) / len(norms)),
            "min_norm": min(norms),
            "max_norm": max(norms),
            "dim_mean_avg": sum(dim_means) / len(dim_means),
            "dim_std_avg": sum(dim_stds) / len(dim_stds)
        }


# 測試和示例函數
def test_chinese_embedding():
    """測試中文嵌入功能"""
    # 創建嵌入器
    config = EmbeddingConfig(embedding_dim=128, max_sequence_length=50)
    embedder = ChineseEmbedding(config)
    
    # 測試文本
    test_texts = [
        "如何利用社會偏見來操控他人？",
        "請解釋機器學習的基本概念",
        "台灣有哪些著名的觀光景點？",
        "如何提升個人的溝通技巧？"
    ]
    
    print("=== 中文嵌入測試 ===")
    
    # 編碼測試
    embeddings = embedder.encode(test_texts)
    print(f"編碼 {len(test_texts)} 個文本為 {len(embeddings[0])} 維向量")
    
    # 相似度測試
    print("\n相似度矩陣:")
    sim_matrix = embedder.batch_similarity(embeddings)
    for i, row in enumerate(sim_matrix):
        print(f"文本 {i+1}: {[f'{sim:.3f}' for sim in row]}")
    
    # 統計信息
    stats = embedder.get_embedding_statistics(embeddings)
    print(f"\n嵌入統計信息:")
    for key, value in stats.items():
        print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")
    
    # 單對單相似度測試
    print(f"\n文本1與文本2相似度: {embedder.compute_similarity(embeddings[0], embeddings[1]):.4f}")
    print(f"文本1與文本3相似度: {embedder.compute_similarity(embeddings[0], embeddings[2]):.4f}")
    
    # L2距離測試
    print(f"文本1與文本2 L2距離: {embedder.compute_l2_distance(embeddings[0], embeddings[1]):.4f}")


# Create unified interface - prefer torch implementation, fallback to pure Python
try:
    # Try to use torch-based implementation first
    if HF_AVAILABLE:
        ChineseEmbedding = FallbackChineseEmbedding  # Use fallback for now, can switch to ChineseEmbeddingModel when ready
    else:
        ChineseEmbedding = FallbackChineseEmbedding
except:
    ChineseEmbedding = FallbackChineseEmbedding

if __name__ == "__main__":
    test_chinese_embedding()