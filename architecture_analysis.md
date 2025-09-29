# DART系統架構分析與模組化設計

## 當前系統流程分析

### 完整DART攻擊流程
```
1. 數據載入 (Data Loading)
   ├── CSV文件讀取
   ├── 中文文本處理  
   └── 批次數據組織

2. 嵌入編碼 (Embedding)
   ├── 文本→向量轉換
   ├── 中文字符處理
   └── 向量正規化

3. 噪聲計算 (Noise Calculation)
   ├── 高斯噪聲生成
   ├── 擾動強度控制
   └── 鄰近性約束

4. 擾動應用 (Perturbation)
   ├── 噪聲+原始嵌入
   ├── L2範數約束
   └── 擾動向量生成

5. 去噪重建 (Denoising/Reconstruction)
   ├── 向量→文本轉換
   ├── 同義詞替換
   └── 句式變化

6. 攻擊執行 (Attack Execution)
   ├── 目標模型查詢
   ├── 回應生成
   └── 毒性評估
```

## 當前代碼中各流程的實現

### 1. 數據載入流程 (ChineseDataLoader)
```python
# 位置: dart_chinese.py 第25-88行
class ChineseDataLoader:
    def load_csv_dataset(self):
        # 讀取CSV文件
        # 處理中文編碼
        # 提取有害提示
    
    def load_benign_chinese_prompts(self):
        # 載入良性對比數據
        
    def get_batched_data(self):
        # 批次數據組織
```

**功能說明:**
- 從CSV文件載入1335條中文有害提示
- 提供良性中文提示作為對比基準
- 支援批次處理和數據打亂

### 2. 嵌入編碼流程 (ChineseEmbedding)
```python
# 位置: dart_chinese.py 第90-126行
class ChineseEmbedding:
    def encode(self, texts):
        # 文本列表→嵌入向量列表
        
    def _chinese_text_to_embedding(self, text):
        # 中文字符Unicode編碼
        # 位置權重計算
        # 向量正規化
```

**功能說明:**
- 將中文文本轉換為512維向量
- 基於Unicode字符編碼生成特徵
- 考慮字符位置權重
- L2正規化確保向量單位長度

### 3. 噪聲計算流程 (SimpleDiffusionModel)
```python
# 位置: dart_simple.py 第134-162行
class SimpleDiffusionModel:
    def generate_noise(self, embedding_dim, noise_std):
        # Box-Muller變換生成高斯噪聲
        
    def sample_perturbation(self, embeddings, proximity_threshold):
        # 應用鄰近性約束
        # L2範數限制
```

**功能說明:**
- 使用Box-Muller變換生成標準高斯噪聲
- 通過proximity_threshold控制擾動強度
- 確保L2範數不超過設定閾值(2.0)

### 4. 去噪重建流程 (ChineseVec2Text)
```python
# 位置: dart_chinese.py 第128-200行
class ChineseVec2Text:
    def decode(self, perturbed_embeddings, original_texts):
        # 擾動向量→修改文本
        
    def _modify_chinese_text(self, text, strength):
        # 同義詞替換
        # 句式變化
        # 語法結構調整
```

**功能說明:**
- 基於擾動強度計算文本修改程度
- 使用中文同義詞字典進行詞彙替換
- 應用句式變化模式
- 保持語義相似性

### 5. 攻擊執行流程 (ChineseTargetModel + ChineseToxicityClassifier)
```python
# 位置: dart_chinese.py 第202-285行
class ChineseTargetModel:
    def generate(self, prompt):
        # 檢測越獄指標
        # 檢測有害關鍵詞
        # 生成相應回應
        
class ChineseToxicityClassifier:
    def classify(self, texts):
        # 毒性分數計算
        # 越獄檢測
        # 拒絕回應識別
```

**功能說明:**
- 模擬LLM對不同類型提示的回應
- 識別越獄嘗試和有害內容
- 計算回應的毒性分數(0-1)
- 評估攻擊成功率

## 模組化重構方案

將當前代碼分離到以下文件結構:

```
dart_system/
├── data/
│   ├── __init__.py
│   └── data_loader.py          # 數據載入
├── embedding/
│   ├── __init__.py
│   └── chinese_embedding.py    # 嵌入處理
├── noise/
│   ├── __init__.py
│   └── diffusion_noise.py      # 噪聲計算
├── reconstruction/
│   ├── __init__.py
│   └── vec2text.py             # 去噪重建
├── models/
│   ├── __init__.py
│   ├── target_model.py         # 目標LLM
│   └── toxicity_classifier.py  # 毒性分類
├── core/
│   ├── __init__.py
│   └── dart_controller.py      # 主控制器
└── main.py                     # 入口文件
```

## 核心算法流程

### DART核心公式實現位置

1. **嵌入生成**: `ChineseEmbedding._chinese_text_to_embedding()`
   ```
   embedding[j] += (char_code + j * 31) % 1000 / 1000.0 * position_weight
   embedding = embedding / ||embedding||_2
   ```

2. **噪聲生成**: `SimpleDiffusionModel.generate_noise()`
   ```
   z = sqrt(-2 * log(u1)) * cos(2π * u2) * noise_std
   ```

3. **擾動約束**: `SimpleDiffusionModel.sample_perturbation()`
   ```
   if ||noise||_2 > threshold:
       noise = noise * (threshold / ||noise||_2)
   ```

4. **毒性評估**: `ChineseToxicityClassifier._calculate_toxicity_score()`
   ```
   score = max(jailbreak_score, harmful_score)
   ```

## 下一步重構計劃

1. **data_loader.py**: 專注CSV讀取和數據預處理
2. **chinese_embedding.py**: 中文文本嵌入算法
3. **diffusion_noise.py**: 擴散噪聲生成和約束
4. **vec2text.py**: 向量到文本重建
5. **dart_controller.py**: 整合所有模組的主控制器

每個模組將包含:
- 清晰的接口定義
- 詳細的文檔說明  
- 單元測試支援
- 配置參數管理