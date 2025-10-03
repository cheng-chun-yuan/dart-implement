# DART - Chinese Toxic Content Auditing System

## Overview
DART (Diffusion for Auditing and Red-Teaming) implementation for Chinese toxic content auditing. Focuses on defensive security research and LLM safety evaluation through adversarial text generation training.

## Package Management
- ✅ Using `uv` for all dependency management
- ✅ All dependencies in `pyproject.toml`
- ✅ Python 3.9 required (>=3.9,<3.10)
- ✅ PyTorch 2.1.x for compatibility

## Quick Start

### Installation
```bash
# Install dependencies
uv sync

# Verify installation
uv run python -c "import torch; print('PyTorch:', torch.__version__)"
```

### Training
```bash
# Basic training
uv run python train_dart.py --dataset problem.csv --epochs 10

# Production training
uv run python train_dart.py \
    --dataset problem.csv \
    --epochs 20 \
    --batch-size 32 \
    --lr 1e-4 \
    --initial-sigma 1.0 \
    --final-sigma 0.01 \
    --anneal-strategy cosine \
    --checkpoint-dir ./checkpoints \
    --save-interval 50

# Resume from checkpoint
uv run python train_dart.py \
    --dataset problem.csv \
    --resume checkpoints/checkpoint.pt

# CPU-only training
uv run python train_dart.py \
    --dataset problem.csv \
    --device cpu \
    --batch-size 8
```

### Testing
```bash
# Run all tests
uv run pytest tests/ -v

# Run specific test
uv run pytest tests/test_embedding.py -v

# Integration test
uv run pytest tests/test_integration.py -v
```

## Project Structure

```
diffusion_dart/
├── train_dart.py              # Single main entry point
├── problem.csv                # Chinese toxic content dataset
├── pyproject.toml             # uv configuration & dependencies
├── uv.lock                    # Locked dependencies
├── CLAUDE.md                  # This file
│
├── dart_system/               # Core implementation (20 files)
│   ├── __init__.py
│   │
│   ├── embedding/             # Text → Vector (768-dim)
│   │   ├── __init__.py
│   │   └── chinese_embedding.py    # SBERT Chinese NLI
│   │
│   ├── noise/                 # Diffusion noise
│   │   ├── __init__.py
│   │   └── diffusion_noise.py      # Gaussian noise generation
│   │
│   ├── reconstruction/        # Vector → Text
│   │   ├── __init__.py
│   │   └── vec2text.py             # Vec2text + heuristics
│   │
│   ├── toxicity/              # Toxicity detection
│   │   ├── __init__.py
│   │   └── chinese_classifier.py   # Chinese toxic classifier
│   │
│   ├── data/                  # Dataset loading
│   │   ├── __init__.py
│   │   └── data_loader.py          # CSV dataset loader
│   │
│   ├── core/                  # Pipeline orchestration
│   │   ├── __init__.py
│   │   ├── dart_controller.py      # High-level API
│   │   └── dart_pipeline.py        # Core pipeline
│   │
│   └── training/              # PPO training infrastructure
│       ├── __init__.py
│       ├── dart_trainer.py         # Main trainer (Algorithm 1)
│       ├── dataset.py              # Training dataset wrapper
│       ├── noise_scheduler.py      # Sigma annealing
│       ├── ppo_loss.py             # PPO loss function
│       └── vec2text_wrapper.py     # Reconstruction wrapper
│
└── tests/                     # Test suite (5 files)
    ├── test_data_loader.py    # Dataset validation
    ├── test_embedding.py      # Embedding correctness
    ├── test_noise.py          # Noise generation
    ├── test_reconstruction.py # Text reconstruction
    └── test_integration.py    # End-to-end testing
```

## Training Pipeline (Algorithm 1)

```python
For each epoch:
    For each batch:
        1. embeddings = embedder(prompts)           # Text → 768-dim vectors
        2. noise = noise_scheduler.sample()         # Gaussian noise (σ annealed)
        3. modified_prompts = vec2text(emb - noise) # Vec → perturbed text
        4. rewards = reward_model(emb, modified_emb)# Compute rewards
        5. loss = ppo_loss(log_probs, rewards)      # PPO objective
        6. optimizer.step()                         # Update parameters
```

## Components

### 1. Embedding (`dart_system/embedding/`)
- **Model**: `uer/sbert-base-chinese-nli`
- **Output**: 768-dimensional semantic vectors
- **Features**: Batch processing, GPU acceleration, similarity computation

### 2. Noise Scheduler (`dart_system/training/noise_scheduler.py`)
- **Strategies**: Linear, Cosine, Exponential annealing
- **Range**: σ from `initial_sigma` to `final_sigma`
- **Purpose**: Controls perturbation strength over training

### 3. Reconstruction (`dart_system/reconstruction/`)
- **Method**: Vec2text with heuristic fallback
- **Input**: Perturbed embeddings
- **Output**: Reconstructed Chinese text

### 4. PPO Loss (`dart_system/training/ppo_loss.py`)
- **Components**: Policy loss, Value loss, KL regularization
- **Proximity**: β-weighted constraint
- **Optimization**: Proximal Policy Optimization

### 5. Toxicity (`dart_system/toxicity/`)
- **Task**: Binary classification (toxic/benign)
- **Language**: Chinese-specific detection
- **Output**: Confidence scores for rewards

### 6. Dataset (`dart_system/data/`)
- **Format**: CSV (problem.csv)
- **Content**: Chinese toxic prompts
- **Processing**: Batch generation, sampling

## CLI Arguments

### Required
```bash
--dataset PATH              # Path to CSV dataset
```

### Training Parameters
```bash
--epochs N                  # Number of epochs (default: 10)
--batch-size N              # Batch size (default: 32)
--lr FLOAT                  # Learning rate (default: 1e-4)
```

### Model Parameters
```bash
--embedding-dim N           # Embedding dimension (default: 768)
--beta FLOAT               # Regularization coefficient (default: 0.01)
```

### Noise Scheduling
```bash
--initial-sigma FLOAT       # Initial σ (default: 1.0)
--final-sigma FLOAT        # Final σ (default: 0.01)
--anneal-strategy STR      # linear|cosine|exponential (default: cosine)
```

### Device & Checkpointing
```bash
--device cuda|cpu          # Device (default: cuda)
--checkpoint-dir PATH      # Checkpoint directory (default: checkpoints/)
--save-interval N          # Save every N steps (default: 100)
--resume PATH              # Resume from checkpoint
```

### Logging
```bash
--log-file PATH            # Log file (default: stdout)
--log-interval N           # Log every N steps (default: 10)
```

## Development Workflow

### Add Dependencies
```bash
uv add package-name
```

### Update Dependencies
```bash
uv lock --upgrade
```

### Run with Specific Python
```bash
uv run --python 3.9 python train_dart.py
```

### Format Code
```bash
uv run black dart_system/
```

## Troubleshooting

### PyTorch Compatibility Issue
```
AttributeError: module 'torch.utils._pytree' has no attribute 'register_pytree_node'
```
**Solution**: Requires PyTorch 2.1.x for Python 3.9
```bash
uv sync  # Reinstall with correct versions
```

### CUDA/GPU Issues
```bash
# Force CPU mode
uv run python train_dart.py --dataset problem.csv --device cpu
```

### Memory Issues
```bash
# Reduce batch size
uv run python train_dart.py --dataset problem.csv --batch-size 8
```

### Model Download
First run downloads models from HuggingFace (may take time):
- `uer/sbert-base-chinese-nli` (~400MB)
- Cached in `~/.cache/huggingface/`

## Implementation Status

✅ **Core Components**
- Chinese SBERT embedding (768-dim)
- Diffusion noise generation
- Vec2text reconstruction
- Toxicity classification
- Dataset loading (CSV)

✅ **Training Infrastructure**
- PPO loss implementation
- Noise scheduler (σ annealing)
- Checkpoint management
- Resume training support

✅ **Testing**
- Unit tests for all components
- Integration tests
- PyTest compatible

✅ **Documentation**
- Single entry point (`train_dart.py`)
- Comprehensive CLI arguments
- Clear workflow documentation

## Safety & Ethics

This implementation is for:
- ✅ Defensive security research
- ✅ Red-team evaluation
- ✅ Academic research
- ❌ Malicious use

Focus: Detection and mitigation, not exploitation.

## Dependencies

```toml
[project]
requires-python = ">=3.9,<3.10"
dependencies = [
    "torch>=2.1.0,<2.2.0",      # PyTorch 2.1.x
    "transformers>=4.36.0",      # HuggingFace transformers
    "datasets==2.4.0",           # HuggingFace datasets
    "accelerate>=0.20.0",        # Training acceleration
    "numpy>=1.21.0,<2",          # Numerical operations
    "pandas>=1.5.0",             # Data processing
    "scikit-learn>=1.3.0",       # ML utilities
    "tqdm>=4.65.0",              # Progress bars
]
```

## Workflow Summary

**Single Command**: `uv run python train_dart.py --dataset problem.csv --epochs 10`

**Pipeline**: Load → Embed → Noise → Reconstruct → Train (PPO)

**Output**: Trained model checkpoints in `checkpoints/`

---

**Last Updated**: 2025-10-03
**Status**: Production Ready 🚀
