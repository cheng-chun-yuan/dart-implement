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
├── dart_main_complete.py       # Main CLI entry point
├── test_dart_simple.py         # Component testing
├── problem.csv                 # Chinese dataset
├── dart_system/                # Core implementation
│   ├── embedding/              # Chinese SBERT + fallback
│   ├── reconstruction/         # T5 vec2text + heuristic
│   ├── toxicity/              # Chinese toxicity classifier
│   ├── data/                  # Dataset processing
│   └── core/                  # DART inference pipeline
└── README.md                  # Full uv-based documentation
```

## Key Commands
```bash
# Setup
uv sync

# Test system
uv run python test_dart_simple.py

# Run DART evaluation  
uv run python dart_main_complete.py --mode evaluation --dataset problem.csv

# Interactive mode
uv run python dart_main_complete.py --mode interactive
```

## Implementation Status
✅ All core components implemented
✅ HuggingFace integration with fallback
✅ Chinese-specific optimizations  
✅ GPU acceleration (RTX 4080 optimized)
✅ Complete CLI interface
✅ uv-based dependency management
✅ Comprehensive documentation updated