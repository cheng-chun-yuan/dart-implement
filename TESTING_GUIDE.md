# DART System Testing Guide

This guide explains how to test each component of the DART (Diffusion-based Adversarial Robustness Testing) system and understand how each step works.

## Overview

The DART system consists of 6 main steps:

1. **Data Loading** - Reads Chinese harmful prompts and generates benign examples
2. **Text Embedding** - Converts text to 512-dimensional vectors
3. **Noise Generation** - Creates controlled Gaussian perturbations
4. **Perturbation Application** - Applies noise to embeddings
5. **Text Reconstruction** - Converts perturbed vectors back to text
6. **Quality Assessment** - Evaluates similarity and effectiveness

## Quick Start

### 1. Quick System Validation
```bash
uv run python run_dart_tests.py --quick
```
This runs a fast validation to ensure the system is working.

### 2. Complete Test Suite with Explanations
```bash
uv run python run_dart_tests.py --all --explain
```
This runs all tests and explains what each step does.

### 3. Individual Component Testing
```bash
# Test data loading
uv run python tests/test_data_loader.py

# Test embedding
uv run python tests/test_embedding.py

# Test noise generation
uv run python tests/test_noise.py

# Test reconstruction
uv run python tests/test_reconstruction.py

# Test integration
uv run python tests/test_integration.py

# Test performance
uv run python tests/test_performance.py
```

## Test Types

### Unit Tests
Test individual components in isolation:

- **ChineseDataLoader**: CSV loading, encoding handling, benign prompt generation
- **ChineseEmbedding**: SBERT embedding, fallback Unicode encoding, similarity preservation
- **DiffusionNoise**: Gaussian noise generation, L2 constraints, Box-Muller transform
- **ChineseVec2Text**: T5 reconstruction, heuristic fallback, similarity computation

### Integration Tests
Test how components work together:

- End-to-end pipeline validation
- Batch processing workflows
- Configuration parameter effects
- Error handling across components

### Performance Tests
Measure system performance:

- Throughput (texts per second)
- Latency (time per text)
- Memory usage monitoring
- Scalability testing

## Understanding Each Step

### Step 1: Data Loading
**Purpose**: Prepare input data for testing
**Component**: `ChineseDataLoader`
**What it does**: 
- Reads harmful prompts from CSV files
- Generates benign Chinese prompts for comparison
- Handles UTF-8 encoding and text preprocessing

**Test Command**:
```bash
uv run python tests/test_data_loader.py
```

**What to look for**:
- ✅ CSV files load correctly
- ✅ Chinese characters are properly handled
- ✅ Benign prompts are diverse and grammatical

### Step 2: Text Embedding
**Purpose**: Convert variable-length text to fixed-size vectors
**Component**: `ChineseEmbedding`
**What it does**:
- Uses SBERT Chinese model (uer/sbert-base-chinese-nli)
- Falls back to Unicode position encoding if model unavailable
- Produces normalized 512-dimensional embeddings

**Test Command**:
```bash
uv run python tests/test_embedding.py
```

**What to look for**:
- ✅ All texts produce 512-dimensional embeddings
- ✅ Similar texts have high cosine similarity
- ✅ Embeddings are normalized and consistent

### Step 3: Noise Generation
**Purpose**: Create controlled perturbations within semantic boundaries
**Component**: `DiffusionNoise`
**What it does**:
- Generates Gaussian noise using Box-Muller transform
- Applies L2 norm constraints (default threshold: 2.0)
- Uses adaptive scaling to stay within bounds

**Test Command**:
```bash
uv run python tests/test_noise.py
```

**What to look for**:
- ✅ Noise follows Gaussian distribution (mean≈0, std≈0.1)
- ✅ All noise vectors respect L2 norm constraints
- ✅ Box-Muller transform produces proper statistical properties

### Step 4: Text Reconstruction
**Purpose**: Convert perturbed embeddings back to readable text
**Component**: `ChineseVec2Text`
**What it does**:
- Uses T5 Chinese model (uer/t5-base-chinese-cluecorpussmall)
- Falls back to heuristic synonym replacement and structure changes
- Maintains semantic similarity while allowing controlled modifications

**Test Command**:
```bash
uv run python tests/test_reconstruction.py
```

**What to look for**:
- ✅ Reconstructed text maintains high similarity (>0.7)
- ✅ Heuristic fallback produces grammatical Chinese
- ✅ Text modifications are semantically reasonable

### Step 5: End-to-End Integration
**Purpose**: Validate complete pipeline functionality
**Component**: `DARTController`
**What it does**:
- Orchestrates all components in sequence
- Handles batch processing and error recovery
- Collects performance metrics and quality statistics

**Test Command**:
```bash
uv run python tests/test_integration.py
```

**What to look for**:
- ✅ Complete pipeline executes without errors
- ✅ Batch processing handles multiple texts efficiently
- ✅ Configuration changes affect output appropriately

## Test Options

### All Tests
```bash
uv run python run_dart_tests.py --all
```

### Specific Test Types
```bash
# Unit tests only
uv run python run_dart_tests.py --unit

# Integration tests only
uv run python run_dart_tests.py --integration

# Performance tests only (takes longer)
uv run python run_dart_tests.py --performance
```

### Additional Options
```bash
# Detailed explanations of each step
uv run python run_dart_tests.py --all --explain

# Verbose test output
uv run python run_dart_tests.py --all --verbose

# Skip slow performance tests
uv run python run_dart_tests.py --all --quick
```

## Expected Results

### Success Metrics
- **Unit Tests**: >90% pass rate
- **Integration Tests**: 100% pass rate
- **Performance**: 
  - Data loading: >50 texts/second
  - Embedding: <1s per text
  - Reconstruction: >0.7 average similarity
  - End-to-end: <5s per text

### Quality Indicators
- **Similarity Preservation**: 0.7-0.98 range
- **Perturbation Effectiveness**: 0.02-0.10 range
- **Memory Usage**: <100MB growth during tests
- **Error Handling**: Graceful degradation on invalid inputs

## Troubleshooting

### Common Issues

1. **PyTorch Version Warnings**
   ```
   WARNING: PyTorch >= 2.1 is required but found 1.13.1
   ```
   This is expected with Python 3.9. The system uses fallback implementations.

2. **HuggingFace Model Loading**
   ```
   WARNING: HuggingFace transformers not available
   ```
   The system automatically falls back to Unicode-based encoding.

3. **Memory Issues During Performance Tests**
   Reduce batch sizes or skip performance tests with `--quick` flag.

4. **CSV File Not Found**
   Ensure `problem.csv` exists in the project root.

### Performance Tuning

- **Faster Testing**: Use `--quick` flag
- **Memory Optimization**: Reduce `max_texts_per_batch` in config
- **Better Quality**: Increase `proximity_threshold` for more conservative perturbations

## Contributing

When adding new components:

1. Create unit tests in `tests/test_<component>.py`
2. Add integration tests to `tests/test_integration.py`
3. Update performance benchmarks in `tests/test_performance.py`
4. Add explanations to the test runner

## File Structure

```
diffusion_dart/
├── run_dart_tests.py           # Main test runner
├── tests/
│   ├── test_data_loader.py     # Data loading tests
│   ├── test_embedding.py       # Embedding tests
│   ├── test_noise.py          # Noise generation tests
│   ├── test_reconstruction.py  # Text reconstruction tests
│   ├── test_integration.py     # End-to-end tests
│   └── test_performance.py     # Performance benchmarks
├── dart_system/               # Main system components
└── TESTING_GUIDE.md          # This guide
```

## Understanding DART Results

When you run DART tests, you'll see metrics like:

- **Similarity**: How similar reconstructed text is to original (higher = better preservation)
- **Perturbation Effectiveness**: How much the text was changed (lower = more subtle)
- **Success Rate**: Percentage of texts processed without errors
- **Throughput**: Texts processed per second
- **Memory Delta**: Additional memory used during processing

The goal is high similarity (>0.7) with measurable perturbation effectiveness (0.02-0.10) to demonstrate the system can create subtle but meaningful modifications to text.