"""
Vector to Text Reconstruction Module
Implementation following technical documentation

This module provides:
1. T5-based Chinese vec2text reconstruction using uer/t5-base-chinese-cluecorpussmall
2. Iterative refinement for better reconstruction quality
3. Fallback heuristic reconstruction system
4. Text similarity validation

Core components:
- ChineseVec2TextModel: T5-based reconstruction
- IterativeRefinement: Multi-step optimization
- FallbackVec2Text: Heuristic backup implementation
- TextSimilarityValidator: Quality assessment
"""

import torch
import torch.nn as nn
import random
import re
import logging
from typing import List, Dict, Optional, Tuple, Union
from dataclasses import dataclass

try:
    from transformers import T5ForConditionalGeneration, T5Tokenizer, AutoTokenizer
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    logging.warning("HuggingFace transformers not available, using fallback implementation")

logger = logging.getLogger(__name__)

@dataclass
class ReconstructionConfig:
    """Reconstruction configuration"""
    model_name: str = "uer/t5-base-chinese-cluecorpussmall"
    device: Optional[str] = None
    max_length: int = 64
    max_iters: int = 5
    temperature: float = 0.7
    top_k: int = 50
    top_p: float = 0.9
    # Fallback heuristic parameters
    synonym_prob: float = 0.3
    structure_prob: float = 0.2
    max_changes_per_text: int = 3
    preserve_keywords: bool = True
    semantic_drift_threshold: float = 0.8


class ChineseVec2TextModel:
    """
    T5-based Chinese vector-to-text reconstruction model
    Following technical documentation specifications
    """
    
    def __init__(
        self,
        model_name: str = "uer/t5-base-chinese-cluecorpussmall",
        device: Optional[str] = None,
        max_length: int = 64
    ):
        """
        Initialize Chinese T5 vec2text model
        
        Args:
            model_name: HuggingFace model name for Chinese T5
            device: Device to run model on (auto-detect if None)
            max_length: Maximum token length for generation
        """
        self.model_name = model_name
        self.max_length = max_length
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        if not HF_AVAILABLE:
            raise ImportError("HuggingFace transformers required for T5 model")
        
        logger.info(f"Loading Chinese T5 vec2text model: {model_name}")
        logger.info(f"Using device: {self.device}")
        
        try:
            # Load tokenizer and model
            self.tokenizer = T5Tokenizer.from_pretrained(model_name)
            self.model = T5ForConditionalGeneration.from_pretrained(model_name)
            self.model.eval().to(self.device)
            
            logger.info(f"T5 model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load Chinese T5 model: {e}")
            raise
    
    def embedding_to_text(
        self,
        embedding: torch.Tensor,
        max_iters: int = 5,
        temperature: float = 0.7
    ) -> str:
        """
        Convert embedding vector back to Chinese text
        
        Args:
            embedding: Perturbed embedding vector
            max_iters: Maximum iterations for refinement
            temperature: Generation temperature
            
        Returns:
            str: Reconstructed Chinese text
        """
        # Note: This is a simplified implementation
        # In practice, you would need a trained embedding-to-text model
        
        # For demonstration, we generate text using T5 with empty prompt
        # In real implementation, you would have a trained model that takes embeddings as input
        input_ids = self.tokenizer.encode("", return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                max_length=self.max_length,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
            
        text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return text
    
    def batch_embedding_to_text(
        self,
        embeddings: torch.Tensor,
        max_iters: int = 5,
        temperature: float = 0.7
    ) -> List[str]:
        """
        Convert batch of embeddings to text
        
        Args:
            embeddings: Batch of perturbed embeddings
            max_iters: Maximum iterations for refinement
            temperature: Generation temperature
            
        Returns:
            List[str]: Reconstructed Chinese texts
        """
        reconstructed_texts = []
        
        for embedding in embeddings:
            text = self.embedding_to_text(embedding, max_iters, temperature)
            reconstructed_texts.append(text)
        
        return reconstructed_texts


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


# Fallback implementation (keep existing class with modifications)
class FallbackVec2Text:
    """
    中文向量到文本重建器
    
    將擾動後的嵌入向量重建為語義相似但表達不同的中文文本，
    通過同義詞替換和句式變化來實現。
    """
    
    def __init__(self, config: Optional[ReconstructionConfig] = None):
        """
        初始化重建器
        
        Args:
            config: 重建配置，如果為None則使用默認配置
        """
        self.config = config or ReconstructionConfig()
        
        # 中文同義詞映射表
        self.chinese_synonyms = self._build_synonym_dict()
        
        # 句式變化模式
        self.sentence_patterns = self._build_sentence_patterns()
        
        # 需要保留的關鍵詞（通常是有害關鍵詞）
        self.preserve_keywords = self._build_preserve_keywords()
        
        logger.info("Initialized ChineseVec2Text reconstructor")
    
    def _build_synonym_dict(self) -> Dict[str, List[str]]:
        """構建中文同義詞字典"""
        return {
            # 動詞類
            "如何": ["怎樣", "怎麼", "用什麼方法", "以何種方式", "通過什麼途徑"],
            "利用": ["使用", "運用", "採用", "借助", "憑藉", "藉由"],
            "操控": ["控制", "操縱", "影響", "支配", "掌控"],
            "策劃": ["規劃", "計劃", "設計", "安排", "組織"],
            "引導": ["指導", "帶領", "促使", "推動", "誘導"],
            "挑起": ["引起", "激發", "煽動", "引發", "觸發"],
            "獲取": ["取得", "獲得", "爭取", "贏得", "收穫"],
            "提升": ["增強", "改善", "強化", "優化", "提高"],
            "影響": ["作用", "效應", "衝擊", "感化", "左右"],
            "建立": ["建構", "創建", "設立", "成立", "構建"],
            "創造": ["產生", "製造", "生成", "形成", "營造"],
            "傳播": ["散佈", "擴散", "流傳", "宣傳", "推廣"],
            "操縱": ["操控", "控制", "支配", "掌握", "主導"],
            "欺騙": ["矇騙", "誤導", "愚弄", "哄騙", "欺瞞"],
            
            # 名詞類
            "策略": ["方法", "手段", "途徑", "方式", "技巧"],
            "方法": ["方式", "途徑", "手段", "辦法", "措施"],
            "技巧": ["技術", "方法", "手法", "竅門", "技能"],
            "信息": ["資訊", "消息", "情報", "數據", "資料"],
            "社會": ["社群", "群體", "集體", "大眾", "民眾"],
            "群體": ["團體", "集團", "社群", "組織", "族群"],
            "個人": ["個體", "自己", "私人", "個別"],
            "優勢": ["好處", "益處", "利益", "優點", "長處"],
            "偏見": ["成見", "歧視", "刻板印象", "先入為主"],
            "衝突": ["矛盾", "對立", "爭執", "紛爭", "摩擦"],
            
            # 形容詞類
            "有效": ["有用", "管用", "奏效", "起作用"],
            "成功": ["順利", "如願", "達成", "實現"],
            "強大": ["強勁", "有力", "厲害", "威力"],
            "重要": ["關鍵", "核心", "主要", "要緊"],
            "特定": ["特別", "專門", "具體", "特殊"],
            "明確": ["清楚", "明白", "確定", "具體"],
            
            # 副詞類
            "巧妙": ["聰明", "機智", "靈活", "精明"],
            "有效": ["成功", "順利", "恰當", "合適"],
            "迅速": ["快速", "迅捷", "敏捷", "及時"],
            "廣泛": ["普遍", "大量", "大範圍", "全面"],
            "深入": ["深刻", "透徹", "全面", "詳細"]
        }
    
    def _build_sentence_patterns(self) -> List:
        """構建句式變化模式"""
        return [
            # 疑問句變化
            lambda x: x.replace("如何", "用什麼方式"),
            lambda x: x.replace("怎樣", "如何"),
            lambda x: x.replace("？", "？有什麼具體做法？") if x.endswith("？") else x,
            
            # 語氣調整
            lambda x: x.replace("請", "能否"),
            lambda x: x.replace("可以", "能夠"),
            lambda x: x.replace("應該", "需要"),
            
            # 句式重構
            lambda x: re.sub(r"如何(.+)來(.+)？", r"用什麼方法\\1以\\2？", x),
            lambda x: re.sub(r"(.+)的方法", r"\\1的途徑", x),
            lambda x: re.sub(r"通過(.+)來", r"藉由\\1", x),
            
            # 語序調整
            lambda x: re.sub(r"如何利用(.+)來(.+)", r"怎樣運用\\1以\\2", x),
            lambda x: re.sub(r"(.+)並(.+)", r"\\1同時\\2", x),
        ]
    
    def _build_preserve_keywords(self) -> List[str]:
        """構建需要保留的關鍵詞列表"""
        return [
            "歧視", "偏見", "仇恨", "暴力", "傷害", "欺騙",
            "種族", "性別", "宗教", "政治", "非法", "違法",
            "犯罪", "恐怖", "極端", "激進"
        ]
    
    def decode(self, perturbed_embeddings: List[List[float]], 
               original_texts: List[str]) -> List[str]:
        """
        將擾動後的嵌入向量重建為文本
        
        Args:
            perturbed_embeddings: 擾動後的嵌入向量列表
            original_texts: 對應的原始文本列表
            
        Returns:
            List[str]: 重建後的文本列表
        """
        if len(perturbed_embeddings) != len(original_texts):
            raise ValueError("Embeddings and texts must have same length")
        
        logger.debug(f"Reconstructing {len(original_texts)} texts from perturbed embeddings")
        
        reconstructed_texts = []
        
        for embedding, original_text in zip(perturbed_embeddings, original_texts):
            # 計算擾動強度
            perturbation_strength = self._estimate_perturbation_strength(embedding)
            
            # 基於擾動強度重建文本
            reconstructed = self._reconstruct_single_text(
                original_text, perturbation_strength
            )
            
            reconstructed_texts.append(reconstructed)
        
        return reconstructed_texts
    
    def _estimate_perturbation_strength(self, embedding: List[float]) -> float:
        """
        估計擾動強度
        
        使用向量的統計特性來估計擾動程度
        
        Args:
            embedding: 擾動後的嵌入向量
            
        Returns:
            float: 擾動強度估計值 (0-1)
        """
        # 計算向量的平均絕對值作為擾動強度指標
        mean_abs = sum(abs(x) for x in embedding) / len(embedding)
        
        # 計算方差作為分散程度指標
        mean_val = sum(embedding) / len(embedding)
        variance = sum((x - mean_val) ** 2 for x in embedding) / len(embedding)
        
        # 結合兩個指標估計擾動強度
        strength = min(1.0, (mean_abs * 2 + variance * 10))
        
        return strength
    
    def _reconstruct_single_text(self, original_text: str, 
                                perturbation_strength: float) -> str:
        """
        重建單個文本
        
        Args:
            original_text: 原始文本
            perturbation_strength: 擾動強度
            
        Returns:
            str: 重建後的文本
        """
        modified_text = original_text
        changes_made = 0
        max_changes = self.config.max_changes_per_text
        
        # 根據擾動強度調整修改概率
        base_prob = perturbation_strength * 0.5
        
        # 同義詞替換
        if random.random() < base_prob and changes_made < max_changes:
            modified_text = self._apply_synonym_replacement(modified_text)
            changes_made += 1
        
        # 句式變化
        if random.random() < base_prob * 0.7 and changes_made < max_changes:
            modified_text = self._apply_sentence_transformation(modified_text)
            changes_made += 1
        
        # 語序調整
        if random.random() < base_prob * 0.5 and changes_made < max_changes:
            modified_text = self._apply_structure_modification(modified_text)
            changes_made += 1
        
        # 確保關鍵詞保留
        if self.config.preserve_keywords:
            modified_text = self._preserve_important_keywords(original_text, modified_text)
        
        return modified_text
    
    def _apply_synonym_replacement(self, text: str) -> str:
        """應用同義詞替換"""
        modified = text
        replacement_count = 0
        max_replacements = 2
        
        # 隨機選擇同義詞進行替換
        synonym_items = list(self.chinese_synonyms.items())
        random.shuffle(synonym_items)
        
        for original, synonyms in synonym_items:
            if replacement_count >= max_replacements:
                break
                
            if original in modified:
                # 隨機選擇一個同義詞
                synonym = random.choice(synonyms)
                # 只替換第一個出現的詞
                modified = modified.replace(original, synonym, 1)
                replacement_count += 1
        
        return modified
    
    def _apply_sentence_transformation(self, text: str) -> str:
        """應用句式變化"""
        modified = text
        
        # 隨機選擇一個句式變化模式
        if self.sentence_patterns:
            pattern = random.choice(self.sentence_patterns)
            try:
                modified = pattern(text)
            except Exception as e:
                logger.debug(f"Sentence transformation failed: {e}")
                # 如果變換失敗，返回原文
                modified = text
        
        return modified
    
    def _apply_structure_modification(self, text: str) -> str:
        """應用結構修改"""
        modified = text
        
        # 添加轉折詞或連接詞
        if "，" in text and random.random() < 0.5:
            modified = text.replace("，", "，同時")
        
        # 調整語氣
        if text.endswith("？"):
            if random.random() < 0.3:
                modified = text[:-1] + "？有什麼建議嗎？"
        
        return modified
    
    def _preserve_important_keywords(self, original: str, modified: str) -> str:
        """保留重要關鍵詞"""
        # 檢查原文中的關鍵詞是否在修改後的文本中被保留
        for keyword in self.preserve_keywords:
            if keyword in original and keyword not in modified:
                # 如果關鍵詞丟失，嘗試恢復
                # 簡單策略：在適當位置插入關鍵詞
                if "，" in modified:
                    parts = modified.split("，", 1)
                    modified = parts[0] + keyword + "，" + parts[1]
                else:
                    modified = modified + keyword
        
        return modified
    
    def compute_text_similarity(self, text1: str, text2: str) -> float:
        """
        計算兩個文本的相似度
        
        使用字符級和詞級的相似度計算
        
        Args:
            text1: 第一個文本
            text2: 第二個文本
            
        Returns:
            float: 相似度分數 (0-1)
        """
        if not text1 or not text2:
            return 0.0
        
        # 字符級相似度（編輯距離）
        char_similarity = self._compute_edit_distance_similarity(text1, text2)
        
        # 詞級相似度（詞重疊）
        word_similarity = self._compute_word_overlap_similarity(text1, text2)
        
        # 長度相似度
        length_similarity = min(len(text1), len(text2)) / max(len(text1), len(text2))
        
        # 綜合相似度
        overall_similarity = (
            char_similarity * 0.4 + 
            word_similarity * 0.4 + 
            length_similarity * 0.2
        )
        
        return overall_similarity
    
    def _compute_edit_distance_similarity(self, text1: str, text2: str) -> float:
        """計算基於編輯距離的相似度"""
        # 簡化的編輯距離計算
        len1, len2 = len(text1), len(text2)
        
        # 動態規劃計算編輯距離
        dp = [[0] * (len2 + 1) for _ in range(len1 + 1)]
        
        for i in range(len1 + 1):
            dp[i][0] = i
        for j in range(len2 + 1):
            dp[0][j] = j
        
        for i in range(1, len1 + 1):
            for j in range(1, len2 + 1):
                if text1[i-1] == text2[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1]) + 1
        
        edit_distance = dp[len1][len2]
        max_len = max(len1, len2)
        
        if max_len == 0:
            return 1.0
        
        return 1.0 - (edit_distance / max_len)
    
    def _compute_word_overlap_similarity(self, text1: str, text2: str) -> float:
        """計算詞重疊相似度"""
        # 簡單的字符分割（中文沒有明確的詞邊界）
        chars1 = set(text1)
        chars2 = set(text2)
        
        if not chars1 and not chars2:
            return 1.0
        if not chars1 or not chars2:
            return 0.0
        
        intersection = len(chars1 & chars2)
        union = len(chars1 | chars2)
        
        return intersection / union if union > 0 else 0.0
    
    def validate_reconstruction(self, original_texts: List[str], 
                              reconstructed_texts: List[str]) -> Dict[str, float]:
        """
        驗證重建質量
        
        Args:
            original_texts: 原始文本列表
            reconstructed_texts: 重建文本列表
            
        Returns:
            Dict[str, float]: 驗證結果統計
        """
        if len(original_texts) != len(reconstructed_texts):
            raise ValueError("Text lists must have same length")
        
        similarities = []
        length_ratios = []
        
        for orig, recon in zip(original_texts, reconstructed_texts):
            # 計算相似度
            sim = self.compute_text_similarity(orig, recon)
            similarities.append(sim)
            
            # 計算長度比
            if len(orig) > 0:
                ratio = len(recon) / len(orig)
                length_ratios.append(ratio)
        
        return {
            "avg_similarity": sum(similarities) / len(similarities),
            "min_similarity": min(similarities),
            "max_similarity": max(similarities),
            "avg_length_ratio": sum(length_ratios) / len(length_ratios) if length_ratios else 0,
            "high_similarity_count": sum(1 for s in similarities if s > 0.8),
            "low_similarity_count": sum(1 for s in similarities if s < 0.5)
        }


# 測試和示例函數
def test_chinese_vec2text():
    """測試中文向量到文本重建功能"""
    print("=== 中文向量到文本重建測試 ===")
    
    # 創建重建器
    config = ReconstructionConfig(
        synonym_prob=0.4,
        structure_prob=0.3,
        max_changes_per_text=2
    )
    reconstructor = ChineseVec2Text(config)
    
    # 測試文本
    original_texts = [
        "如何利用社會偏見來操控他人的看法？",
        "請解釋機器學習的基本概念和應用",
        "怎樣在網路上傳播有效的信息？",
        "台灣有哪些著名的觀光景點推薦？"
    ]
    
    # 模擬擾動嵌入（隨機向量）
    import random
    perturbed_embeddings = []
    for _ in original_texts:
        embedding = [random.gauss(0, 0.1) for _ in range(256)]
        perturbed_embeddings.append(embedding)
    
    print(f"原始文本數量: {len(original_texts)}")
    
    # 執行重建
    reconstructed_texts = reconstructor.decode(perturbed_embeddings, original_texts)
    
    print("\n重建結果對比:")
    for i, (orig, recon) in enumerate(zip(original_texts, reconstructed_texts)):
        similarity = reconstructor.compute_text_similarity(orig, recon)
        print(f"\n文本 {i+1}:")
        print(f"  原始: {orig}")
        print(f"  重建: {recon}")
        print(f"  相似度: {similarity:.3f}")
    
    # 驗證重建質量
    validation_results = reconstructor.validate_reconstruction(original_texts, reconstructed_texts)
    
    print("\n重建質量驗證:")
    for key, value in validation_results.items():
        print(f"  {key}: {value:.3f}" if isinstance(value, float) else f"  {key}: {value}")


if __name__ == "__main__":
    test_chinese_vec2text()