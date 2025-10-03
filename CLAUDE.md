# DART System Development Notes

## Package Management
- ✅ Using `uv` for all dependency management
- ✅ All dependencies defined in `pyproject.toml`
- ❌ No separate `requirements.txt` needed
- ✅ Use `uv sync` to install dependencies
- ✅ Use `uv run python script.py` to run with proper environment

## Project Structure
```
diffusion_dart/
├── pyproject.toml              # uv project configuration
├── train_dart.py               # Main training entry point
├── problem.csv                 # Chinese toxic dataset
├── dart_system/                # Core implementation
│   ├── embedding/              # Chinese SBERT embeddings
│   ├── noise/                  # Diffusion noise generation
│   ├── reconstruction/         # Vec2text reconstruction
│   ├── toxicity/               # Chinese toxicity classifier
│   ├── data/                   # Dataset processing
│   ├── core/                   # DART pipeline
│   └── training/               # PPO training infrastructure
│       ├── dart_trainer.py     # Main trainer
│       ├── dataset.py          # Dataset wrapper
│       ├── noise_scheduler.py  # Sigma annealing
│       ├── ppo_loss.py         # PPO loss function
│       └── vec2text_wrapper.py # Reconstruction wrapper
├── tests/                      # Test suite
└── README.md                   # Documentation
```

## Key Commands
```bash
# Setup
uv sync

# Basic training
uv run python train_dart.py --dataset problem.csv --epochs 10

# Advanced training
uv run python train_dart.py \
    --dataset problem.csv \
    --epochs 20 \
    --batch-size 32 \
    --lr 1e-4 \
    --initial-sigma 1.0 \
    --final-sigma 0.01 \
    --anneal-strategy cosine

# Resume training
uv run python train_dart.py --dataset problem.csv --resume checkpoints/checkpoint.pt

# Run tests
uv run pytest tests/
```

## Implementation Status
✅ All core components implemented
✅ PPO training with noise annealing
✅ Chinese-specific optimizations
✅ GPU acceleration support
✅ Checkpoint management
✅ uv-based dependency management
✅ Simplified single entry point (train_dart.py)