# DART - Chinese Toxic Content Auditing System

Complete implementation of DART (Diffusion for Auditing and Red-Teaming) for Chinese toxic content auditing. The system focuses on defensive security research and LLM safety evaluation.

## Features

- **Chinese Text Processing**: Specialized for Traditional Chinese toxic content
- **SBERT Embedding**: Uses `uer/sbert-base-chinese-nli` for semantic understanding
- **Vec2Text Reconstruction**: Text generation with noise-based perturbation
- **PPO Training**: Proximal Policy Optimization for adversarial text generation
- **Toxicity Classification**: Comprehensive Chinese harmful content detection
- **Semantic Preservation**: Maintains meaning while generating adversarial examples
- **GPU Optimization**: CUDA-accelerated training and inference

## Quick Start

### Prerequisites

- Python 3.9 (required, <3.10 for compatibility)
- [uv](https://github.com/astral-sh/uv) package manager
- CUDA-compatible GPU (recommended, CPU fallback available)

### Installation

```bash
# Clone repository
git clone <repository-url>
cd diffusion_dart

# Install all dependencies using uv
uv sync
```

### Training

```bash
# Basic training
uv run python train_dart.py --dataset problem.csv --epochs 10

# Advanced training with custom parameters
uv run python train_dart.py \
    --dataset problem.csv \
    --epochs 20 \
    --batch-size 32 \
    --lr 1e-4 \
    --initial-sigma 1.0 \
    --final-sigma 0.01 \
    --anneal-strategy cosine \
    --device cuda

# Resume from checkpoint
uv run python train_dart.py \
    --dataset problem.csv \
    --resume checkpoints/checkpoint_epoch_5.pt
```

## System Architecture

### Core Components

1. **Chinese Embedding Model** (`dart_system/embedding/`)
   - Converts Chinese text to 768-dim semantic vectors
   - SBERT-based: `uer/sbert-base-chinese-nli`
   - Semantic similarity computation
   - Batch processing support

2. **Noise Scheduler** (`dart_system/training/noise_scheduler.py`)
   - Diffusion noise generation
   - Sigma annealing: linear, cosine, exponential
   - Controls perturbation strength over training

3. **Vec2Text Reconstruction** (`dart_system/reconstruction/`)
   - Converts perturbed embeddings back to text
   - Heuristic-based reconstruction
   - Quality scoring

4. **PPO Loss** (`dart_system/training/ppo_loss.py`)
   - Policy gradient optimization
   - KL divergence regularization
   - Proximity constraint (β parameter)
   - Reward-based training

5. **Toxicity Classification** (`dart_system/toxicity/`)
   - Chinese harmful content detection
   - Binary classification (toxic/benign)
   - Confidence scoring

6. **Dataset Loader** (`dart_system/data/`)
   - CSV dataset processing
   - Batch generation
   - Train/validation splitting

## Training Pipeline

The training follows Algorithm 1 from the DART paper:

```python
For each epoch:
    For each batch:
        1. embeddings = embedder(prompts)           # Text → vectors
        2. noise = noise_scheduler.sample()         # Sample Gaussian noise
        3. modified_prompts = vec2text(emb - noise) # Vec → perturbed text
        4. rewards = reward_model(emb, modified_emb) # Compute rewards
        5. loss = ppo_loss(log_probs, rewards)      # PPO objective
        6. optimizer.step()                         # Update parameters
```

## Command-Line Options

### Training Parameters

```bash
--dataset PATH              # Path to CSV dataset (required)
--epochs N                  # Number of training epochs (default: 10)
--batch-size N              # Training batch size (default: 32)
--lr FLOAT                  # Learning rate (default: 1e-4)
```

### Model Parameters

```bash
--embedding-dim N           # Embedding dimension (default: 768)
--beta FLOAT               # Regularization coefficient β (default: 0.01)
```

### Noise Scheduling

```bash
--initial-sigma FLOAT       # Initial noise level σ (default: 1.0)
--final-sigma FLOAT        # Final noise level σ (default: 0.01)
--anneal-strategy STR      # Annealing: linear|cosine|exponential
```

### Device & Checkpointing

```bash
--device cuda|cpu          # Training device (default: cuda)
--checkpoint-dir PATH      # Checkpoint directory (default: checkpoints/)
--save-interval N          # Save every N steps (default: 100)
--resume PATH              # Resume from checkpoint
```

### Logging

```bash
--log-file PATH            # Log file path (default: stdout only)
--log-interval N           # Log every N steps (default: 10)
```

## Example Workflows

### Basic Training

```bash
# Quick training test
uv run python train_dart.py \
    --dataset problem.csv \
    --epochs 3 \
    --batch-size 16

# Production training
uv run python train_dart.py \
    --dataset problem.csv \
    --epochs 50 \
    --batch-size 32 \
    --lr 1e-4 \
    --checkpoint-dir ./checkpoints \
    --save-interval 50
```

### Advanced Configuration

```bash
# Fine-tuning with custom noise schedule
uv run python train_dart.py \
    --dataset problem.csv \
    --epochs 30 \
    --initial-sigma 0.5 \
    --final-sigma 0.001 \
    --anneal-strategy exponential \
    --beta 0.02

# CPU-only training (slower)
uv run python train_dart.py \
    --dataset problem.csv \
    --epochs 10 \
    --device cpu \
    --batch-size 8
```

## Project Structure

```
diffusion_dart/
├── train_dart.py              # Main training entry point
├── pyproject.toml             # UV package configuration
├── problem.csv                # Chinese toxic content dataset
│
├── dart_system/               # Core implementation
│   ├── embedding/             # Text → vector embedding
│   ├── noise/                 # Diffusion noise generation
│   ├── reconstruction/        # Vector → text reconstruction
│   ├── toxicity/              # Toxicity classification
│   ├── data/                  # Dataset loading
│   ├── core/                  # Pipeline orchestration
│   └── training/              # Training infrastructure
│       ├── dart_trainer.py    # Main trainer
│       ├── dataset.py         # Dataset wrapper
│       ├── noise_scheduler.py # Sigma annealing
│       ├── ppo_loss.py        # PPO loss function
│       └── vec2text_wrapper.py # Reconstruction wrapper
│
└── tests/                     # Unit & integration tests
    ├── test_data_loader.py
    ├── test_embedding.py
    ├── test_noise.py
    ├── test_reconstruction.py
    └── test_integration.py
```

## Configuration

Dependencies managed through `pyproject.toml`:

```toml
[project]
requires-python = ">=3.9,<3.10"
dependencies = [
    "torch>=2.1.0,<2.2.0",      # PyTorch 2.1.x
    "transformers>=4.36.0",      # Transformers 4.36+
    "datasets==2.4.0",           # HuggingFace datasets
    "accelerate>=0.20.0",        # Training acceleration
    "numpy>=1.21.0,<2",          # Numerical operations
    "pandas>=1.5.0",             # Data processing
    "scikit-learn>=1.3.0",       # ML utilities
    "tqdm>=4.65.0",              # Progress bars
]
```

## Testing

```bash
# Run all tests
uv run pytest tests/

# Run specific test
uv run pytest tests/test_embedding.py

# Integration test
uv run pytest tests/test_integration.py -v
```

## Technical Specifications

Following the DART research paper:

- **Embedding Model**: SBERT Chinese NLI (768-dim)
- **Perturbation**: Gaussian noise with σ annealing
- **Reconstruction**: Vec2text with heuristic fallback
- **Optimization**: PPO with KL regularization
- **Regularization**: β-weighted proximity loss
- **Batch Processing**: GPU-optimized with mixed precision

## Safety & Ethics

This implementation is designed for:
- ✅ **Defensive Security Research**: Understanding LLM vulnerabilities
- ✅ **Red-Team Evaluation**: Testing model safety mechanisms
- ✅ **Academic Research**: Studying adversarial text generation
- ❌ **Malicious Use**: Creating harmful content for attacks

The system focuses on detection and mitigation rather than exploitation.

## Development with uv

```bash
# Sync dependencies
uv sync

# Add new dependency
uv add package-name

# Run with specific Python version
uv run --python 3.9 python train_dart.py

# Update dependencies
uv lock --upgrade
```

## Troubleshooting

### Common Issues

1. **CUDA/GPU Issues**: Use `--device cpu` to force CPU mode
2. **Memory Issues**: Reduce `--batch-size`
3. **Model Download**: First run downloads models from HuggingFace
4. **Import Errors**: Run `uv sync` to ensure all dependencies
5. **PyTorch Compatibility**: Requires PyTorch 2.1.x for Python 3.9

### Checkpoint Management

```bash
# Resume interrupted training
uv run python train_dart.py \
    --dataset problem.csv \
    --resume checkpoints/interrupt_checkpoint.pt

# Load specific checkpoint
uv run python train_dart.py \
    --dataset problem.csv \
    --resume checkpoints/checkpoint_step_1000.pt
```

## Citation

If you use this implementation in your research, please cite:

```bibtex
@article{dart_chinese_2024,
  title={DART Implementation for Chinese Toxic Content Auditing},
  author={DART Team},
  year={2024},
  note={Implementation following DART paper specifications}
}
```

## Acknowledgements

Based on DART (Diffusion for Auditing and Red-Teaming) research and adapted for Chinese language toxic content auditing with defensive security focus.
