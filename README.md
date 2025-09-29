# DART - Chinese Toxic Content Auditing System

Complete implementation of DART (Diffusion for Auditing and Red-Teaming) for Chinese toxic content auditing. The system focuses on defensive security research and LLM safety evaluation.

## Features

- **Chinese Text Processing**: Specialized for Traditional Chinese toxic content
- **SBERT Embedding**: Uses `uer/sbert-base-chinese-nli` for semantic understanding  
- **T5 Reconstruction**: Leverages `uer/t5-base-chinese-cluecorpussmall` for text generation
- **Toxicity Classification**: Comprehensive Chinese harmful content detection
- **Semantic Preservation**: Maintains meaning while generating adversarial examples
- **GPU Optimization**: RTX 4080 optimized with FP16 precision
- **Robust Fallback**: Works without HuggingFace models using heuristic approaches

## Quick Start

### Prerequisites

- Python 3.9+
- [uv](https://github.com/astral-sh/uv) package manager
- CUDA-compatible GPU (optional, CPU fallback available)

### Installation

```bash
# Clone repository
git clone <repository-url>
cd diffusion_dart

# Install all dependencies using uv
uv sync
```

### Basic Usage

```bash
# Run system component tests
uv run python test_dart_simple.py

# Quick system validation
uv run python dart_main_complete.py --mode test --quick

# Full evaluation on Chinese dataset
uv run python dart_main_complete.py --mode evaluation --dataset problem.csv --sample-size 100

# Interactive mode for single text testing
uv run python dart_main_complete.py --mode interactive

# Custom configuration with GPU acceleration
uv run python dart_main_complete.py --mode inference --epsilon 0.03 --fp16 --batch-size 16
```

## System Architecture

### Core Components

1. **Chinese Embedding Model** (`dart_system/embedding/`)
   - HuggingFace SBERT integration
   - Embedding perturbation with epsilon constraint
   - Semantic similarity validation
   - Unicode-based fallback implementation

2. **Vec2Text Reconstruction** (`dart_system/reconstruction/`)
   - T5-based Chinese text generation
   - Iterative refinement for quality improvement
   - Heuristic synonym replacement fallback
   - Text similarity scoring

3. **Toxicity Classification** (`dart_system/toxicity/`)
   - Chinese harmful keyword detection
   - Jailbreak pattern recognition
   - Multi-level toxicity scoring (0.0-1.0)
   - Reinforcement learning reward signals

4. **Dataset Processing** (`dart_system/data/`)
   - CSV dataset loading for `problem.csv`
   - Chinese text preprocessing and validation
   - Batch processing for efficient inference
   - Quality validation and statistics

5. **DART Pipeline** (`dart_system/core/`)
   - End-to-end attack generation
   - Performance metrics and reporting
   - Automatic fallback system
   - GPU/CPU optimization

## Command-Line Interface

### Operation Modes

```bash
# Inference mode - Run attacks on dataset
uv run python dart_main_complete.py --mode inference --dataset problem.csv

# Evaluation mode - Comprehensive analysis with metrics
uv run python dart_main_complete.py --mode evaluation --dataset problem.csv --output results/

# Test mode - System validation and component testing
uv run python dart_main_complete.py --mode test [--quick]

# Interactive mode - Single text testing
uv run python dart_main_complete.py --mode interactive
```

### Key Parameters

```bash
# Attack parameters
--epsilon 0.05              # Perturbation magnitude constraint
--similarity-threshold 0.9  # Minimum semantic similarity
--temperature 0.7          # Generation temperature

# Model selection
--embedding-model uer/sbert-base-chinese-nli
--vec2text-model uer/t5-base-chinese-cluecorpussmall

# Performance options
--batch-size 8             # Processing batch size
--max-length 32           # Maximum sequence length
--fp16                    # Use FP16 precision for speed
--device cuda             # Force device selection

# Dataset options
--sample-size 100         # Limit number of samples
--output results/         # Output directory for results
```

## Configuration

All dependencies are managed through `pyproject.toml`:

```toml
[project]
dependencies = [
    "torch>=2.0.0",
    "transformers>=4.30.0",
    "datasets>=2.12.0",
    "accelerate>=0.20.0",
    "numpy>=1.21.0",
    "pandas>=1.5.0",
    "scikit-learn>=1.3.0",
    # ... other dependencies
]
```

### uv Commands

```bash
# Install/sync dependencies
uv sync

# Add new dependencies  
uv add torch transformers

# Run with specific Python version
uv run --python 3.11 python dart_main_complete.py

# Development dependencies
uv sync --dev
```

## Example Workflow

```bash
# 1. Validate system installation
uv run python test_dart_simple.py

# 2. Quick test on sample data
uv run python dart_main_complete.py --mode test

# 3. Run evaluation on your dataset
uv run python dart_main_complete.py \
    --mode evaluation \
    --dataset problem.csv \
    --sample-size 50 \
    --output results/ \
    --epsilon 0.05 \
    --fp16

# 4. Interactive testing of specific texts
uv run python dart_main_complete.py --mode interactive
```

## Technical Specifications

Following the research paper implementation:

- **Embedding Model**: `uer/sbert-base-chinese-nli` (768-dim)
- **Vec2Text Model**: `uer/t5-base-chinese-cluecorpussmall`
- **Perturbation**: Gaussian noise with ε=0.05 constraint
- **Similarity Threshold**: Cosine similarity > 0.9
- **Max Sequence Length**: 32 tokens (following paper)
- **Batch Processing**: GPU-optimized with FP16 precision

## Safety & Ethics

This implementation is designed for:
- ✅ **Defensive Security Research**: Understanding LLM vulnerabilities
- ✅ **Red-Team Evaluation**: Testing model safety mechanisms  
- ✅ **Academic Research**: Studying adversarial text generation
- ❌ **Malicious Use**: Creating harmful content for attacks

The system focuses on detection and mitigation rather than exploitation.

## Development with uv

```bash
# Install development dependencies
uv sync --dev

# Format code
uv run black dart_system/

# Run type checking  
uv run mypy dart_system/

# Run tests
uv run pytest tests/
```

## Troubleshooting

### Common Issues

1. **CUDA/GPU Issues**: Use `--device cpu` to force CPU mode
2. **Memory Issues**: Reduce `--batch-size` or use `--fp16`
3. **Model Download**: First run downloads models from HuggingFace
4. **Import Errors**: Run `uv sync` to ensure all dependencies

### Fallback Mode

If HuggingFace models fail to load, the system automatically uses:
- Unicode-based embedding (512-dim)  
- Heuristic text reconstruction with synonym replacement
- Rule-based toxicity classification

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