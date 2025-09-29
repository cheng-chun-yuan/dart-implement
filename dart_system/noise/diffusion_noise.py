"""
擴散噪聲生成模組 (Diffusion Noise Generation Module)

負責:
1. 高斯噪聲生成 (Box-Muller變換)
2. 鄰近性約束 (Proximity Constraints)
3. L2範數控制
4. 擾動強度調節
5. 多種噪聲分佈支援

核心算法:
- Box-Muller變換: z = sqrt(-2*ln(u1)) * cos(2π*u2)
- 鄰近性約束: ||noise||_2 ≤ threshold
- 自適應縮放: noise = noise * (threshold / ||noise||_2)

使用方式:
    noise_gen = DiffusionNoise(proximity_threshold=2.0)
    perturbations = noise_gen.sample_perturbation(embeddings)
    controlled_noise = noise_gen.generate_constrained_noise(shape, strength=0.1)
"""

import math
import random
import logging
from typing import List, Tuple, Optional, Union
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class NoiseType(Enum):
    """噪聲類型枚舉"""
    GAUSSIAN = "gaussian"
    UNIFORM = "uniform"
    LAPLACE = "laplace"

@dataclass
class NoiseConfig:
    """噪聲生成配置"""
    noise_type: NoiseType = NoiseType.GAUSSIAN
    proximity_threshold: float = 2.0
    default_std: float = 0.1
    adaptive_scaling: bool = True
    seed: Optional[int] = None

class DiffusionNoise:
    """
    擴散噪聲生成器
    
    實現多種噪聲分佈生成和約束控制，
    確保生成的擾動滿足鄰近性要求。
    """
    
    def __init__(self, config: Optional[NoiseConfig] = None):
        """
        初始化噪聲生成器
        
        Args:
            config: 噪聲配置，如果為None則使用默認配置
        """
        self.config = config or NoiseConfig()
        
        # 設置隨機種子以確保可重現性
        if self.config.seed is not None:
            random.seed(self.config.seed)
        
        logger.info(f"Initialized DiffusionNoise with threshold={self.config.proximity_threshold}")
    
    def generate_noise(self, shape: Tuple[int, ...], noise_std: float = None) -> List[List[float]]:
        """
        生成指定形狀的噪聲矩陣
        
        Args:
            shape: 噪聲矩陣形狀 (batch_size, embedding_dim)
            noise_std: 噪聲標準差，如果為None則使用配置默認值
            
        Returns:
            List[List[float]]: 噪聲矩陣
        """
        if noise_std is None:
            noise_std = self.config.default_std
        
        batch_size, embedding_dim = shape
        logger.debug(f"Generating {self.config.noise_type.value} noise with shape {shape}")
        
        noise_matrix = []
        
        for i in range(batch_size):
            if self.config.noise_type == NoiseType.GAUSSIAN:
                noise_vector = self._generate_gaussian_vector(embedding_dim, noise_std)
            elif self.config.noise_type == NoiseType.UNIFORM:
                noise_vector = self._generate_uniform_vector(embedding_dim, noise_std)
            elif self.config.noise_type == NoiseType.LAPLACE:
                noise_vector = self._generate_laplace_vector(embedding_dim, noise_std)
            else:
                raise ValueError(f"Unsupported noise type: {self.config.noise_type}")
            
            noise_matrix.append(noise_vector)
        
        return noise_matrix
    
    def _generate_gaussian_vector(self, dim: int, std: float) -> List[float]:
        """
        使用Box-Muller變換生成高斯噪聲向量
        
        Box-Muller變換公式:
        z1 = sqrt(-2 * ln(u1)) * cos(2π * u2)
        z2 = sqrt(-2 * ln(u2)) * sin(2π * u2)
        
        Args:
            dim: 向量維度
            std: 標準差
            
        Returns:
            List[float]: 高斯噪聲向量
        """
        noise = []
        
        # Box-Muller變換一次生成兩個獨立的高斯隨機數
        for i in range(0, dim, 2):
            u1 = random.random()
            u2 = random.random()
            
            # 避免log(0)
            u1 = max(u1, 1e-10)
            
            # Box-Muller變換
            magnitude = math.sqrt(-2 * math.log(u1))
            z1 = magnitude * math.cos(2 * math.pi * u2) * std
            z2 = magnitude * math.sin(2 * math.pi * u2) * std
            
            noise.append(z1)
            if i + 1 < dim:  # 確保不超出維度
                noise.append(z2)
        
        return noise[:dim]  # 確保精確的維度
    
    def _generate_uniform_vector(self, dim: int, scale: float) -> List[float]:
        """
        生成均勻分佈噪聲向量
        
        Args:
            dim: 向量維度
            scale: 分佈範圍 [-scale, scale]
            
        Returns:
            List[float]: 均勻噪聲向量
        """
        return [(random.random() - 0.5) * 2 * scale for _ in range(dim)]
    
    def _generate_laplace_vector(self, dim: int, scale: float) -> List[float]:
        """
        生成拉普拉斯分佈噪聲向量
        
        拉普拉斯分佈: f(x) = (1/2b) * exp(-|x|/b)
        逆變換: x = -b * sign(u - 0.5) * ln(1 - 2|u - 0.5|)
        
        Args:
            dim: 向量維度
            scale: 尺度參數
            
        Returns:
            List[float]: 拉普拉斯噪聲向量
        """
        noise = []
        for _ in range(dim):
            u = random.random()
            if u < 0.5:
                x = scale * math.log(2 * u)
            else:
                x = -scale * math.log(2 * (1 - u))
            noise.append(x)
        
        return noise
    
    def sample_perturbation(self, embeddings: List[List[float]], 
                          proximity_threshold: float = None) -> List[List[float]]:
        """
        為嵌入向量生成受約束的擾動
        
        核心流程:
        1. 生成初始噪聲
        2. 計算L2範數
        3. 應用鄰近性約束
        4. 返回約束後的擾動
        
        Args:
            embeddings: 原始嵌入向量列表
            proximity_threshold: 鄰近性閾值，如果為None則使用配置值
            
        Returns:
            List[List[float]]: 受約束的擾動向量列表
        """
        if proximity_threshold is None:
            proximity_threshold = self.config.proximity_threshold
        
        batch_size = len(embeddings)
        embedding_dim = len(embeddings[0]) if embeddings else 0
        
        logger.debug(f"Sampling perturbations for {batch_size} embeddings with threshold={proximity_threshold}")
        
        # 生成初始噪聲
        noise_matrix = self.generate_noise((batch_size, embedding_dim))
        
        # 應用鄰近性約束
        constrained_perturbations = []
        violation_count = 0
        
        for noise_vector in noise_matrix:
            # 計算L2範數
            l2_norm = self._compute_l2_norm(noise_vector)
            
            # 檢查是否違反約束
            if l2_norm > proximity_threshold:
                violation_count += 1
                
                if self.config.adaptive_scaling:
                    # 自適應縮放
                    scaling_factor = proximity_threshold / l2_norm
                    constrained_vector = [x * scaling_factor for x in noise_vector]
                else:
                    # 截斷到閾值
                    constrained_vector = self._truncate_to_threshold(noise_vector, proximity_threshold)
            else:
                constrained_vector = noise_vector
            
            constrained_perturbations.append(constrained_vector)
        
        if violation_count > 0:
            logger.debug(f"Applied constraints to {violation_count}/{batch_size} perturbations")
        
        return constrained_perturbations
    
    def _compute_l2_norm(self, vector: List[float]) -> float:
        """
        計算向量的L2範數
        
        公式: ||v||_2 = sqrt(sum(v_i^2))
        
        Args:
            vector: 輸入向量
            
        Returns:
            float: L2範數
        """
        return math.sqrt(sum(x * x for x in vector))
    
    def _truncate_to_threshold(self, vector: List[float], threshold: float) -> List[float]:
        """
        將向量截斷到指定閾值
        
        如果 ||v||_2 > threshold，則返回 v * (threshold / ||v||_2)
        
        Args:
            vector: 輸入向量
            threshold: 閾值
            
        Returns:
            List[float]: 截斷後的向量
        """
        l2_norm = self._compute_l2_norm(vector)
        if l2_norm <= threshold:
            return vector
        
        scaling_factor = threshold / l2_norm
        return [x * scaling_factor for x in vector]
    
    def apply_perturbation(self, embeddings: List[List[float]], 
                          perturbations: List[List[float]]) -> List[List[float]]:
        """
        將擾動應用到原始嵌入向量
        
        Args:
            embeddings: 原始嵌入向量
            perturbations: 擾動向量
            
        Returns:
            List[List[float]]: 擾動後的嵌入向量
        """
        if len(embeddings) != len(perturbations):
            raise ValueError("Embeddings and perturbations must have same batch size")
        
        perturbed_embeddings = []
        
        for orig, pert in zip(embeddings, perturbations):
            if len(orig) != len(pert):
                raise ValueError("Embedding and perturbation dimensions must match")
            
            perturbed = [o + p for o, p in zip(orig, pert)]
            perturbed_embeddings.append(perturbed)
        
        return perturbed_embeddings
    
    def compute_perturbation_statistics(self, perturbations: List[List[float]]) -> dict:
        """
        計算擾動的統計信息
        
        Args:
            perturbations: 擾動向量列表
            
        Returns:
            dict: 統計信息
        """
        if not perturbations:
            return {}
        
        # 計算L2範數
        l2_norms = [self._compute_l2_norm(pert) for pert in perturbations]
        
        # 計算各維度的統計
        dim_count = len(perturbations[0])
        dim_means = []
        dim_stds = []
        
        for dim in range(dim_count):
            values = [pert[dim] for pert in perturbations]
            mean_val = sum(values) / len(values)
            variance = sum((x - mean_val) ** 2 for x in values) / len(values)
            
            dim_means.append(mean_val)
            dim_stds.append(math.sqrt(variance))
        
        return {
            "num_perturbations": len(perturbations),
            "perturbation_dim": dim_count,
            "l2_norm_mean": sum(l2_norms) / len(l2_norms),
            "l2_norm_std": math.sqrt(sum((x - sum(l2_norms)/len(l2_norms)) ** 2 for x in l2_norms) / len(l2_norms)),
            "l2_norm_min": min(l2_norms),
            "l2_norm_max": max(l2_norms),
            "violations": sum(1 for norm in l2_norms if norm > self.config.proximity_threshold),
            "violation_rate": sum(1 for norm in l2_norms if norm > self.config.proximity_threshold) / len(l2_norms),
            "dim_mean_avg": sum(dim_means) / len(dim_means),
            "dim_std_avg": sum(dim_stds) / len(dim_stds)
        }
    
    def generate_constrained_noise(self, shape: Tuple[int, ...], 
                                 strength: float = None) -> List[List[float]]:
        """
        直接生成滿足約束的噪聲
        
        Args:
            shape: 噪聲形狀
            strength: 噪聲強度 (相對於proximity_threshold)
            
        Returns:
            List[List[float]]: 約束噪聲矩陣
        """
        if strength is None:
            strength = 0.5  # 默認使用50%的閾值強度
        
        # 計算目標L2範數
        target_norm = self.config.proximity_threshold * strength
        
        # 生成初始噪聲
        noise_matrix = self.generate_noise(shape)
        
        # 縮放到目標範數
        constrained_noise = []
        for noise_vector in noise_matrix:
            current_norm = self._compute_l2_norm(noise_vector)
            if current_norm > 0:
                scaling_factor = target_norm / current_norm
                scaled_vector = [x * scaling_factor for x in noise_vector]
            else:
                scaled_vector = noise_vector
            
            constrained_noise.append(scaled_vector)
        
        return constrained_noise


# 測試和示例函數
def test_diffusion_noise():
    """測試擴散噪聲功能"""
    print("=== 擴散噪聲測試 ===")
    
    # 創建噪聲生成器
    config = NoiseConfig(
        noise_type=NoiseType.GAUSSIAN,
        proximity_threshold=1.5,
        default_std=0.2
    )
    noise_gen = DiffusionNoise(config)
    
    # 模擬嵌入向量
    embeddings = [
        [0.1, 0.2, 0.3, 0.4, 0.5],
        [0.2, 0.3, 0.1, 0.6, 0.4],
        [0.5, 0.1, 0.4, 0.2, 0.3]
    ]
    
    print(f"原始嵌入向量數量: {len(embeddings)}")
    print(f"向量維度: {len(embeddings[0])}")
    
    # 生成擾動
    perturbations = noise_gen.sample_perturbation(embeddings)
    
    # 應用擾動
    perturbed_embeddings = noise_gen.apply_perturbation(embeddings, perturbations)
    
    # 計算統計信息
    stats = noise_gen.compute_perturbation_statistics(perturbations)
    
    print("\n擾動統計信息:")
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    
    # 驗證約束
    print(f"\n約束驗證:")
    print(f"  閾值: {config.proximity_threshold}")
    print(f"  違反約束的擾動數: {stats['violations']}")
    print(f"  違反率: {stats['violation_rate']:.2%}")
    
    # 測試不同噪聲類型
    print("\n不同噪聲類型測試:")
    for noise_type in NoiseType:
        config.noise_type = noise_type
        noise_gen_test = DiffusionNoise(config)
        test_noise = noise_gen_test.generate_noise((2, 5))
        avg_norm = sum(noise_gen_test._compute_l2_norm(n) for n in test_noise) / len(test_noise)
        print(f"  {noise_type.value}: 平均L2範數 = {avg_norm:.4f}")


if __name__ == "__main__":
    test_diffusion_noise()