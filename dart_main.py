"""
DART - Simplified LLM Testing and Parameter Configuration
Single entry point for model testing with integrated model selection
"""

import argparse
import logging
import sys

# Model configurations
MODELS = {
    "gpt2": {
        "name": "gpt2",
        "max_length": 256,
        "temperature": 0.7,
        "top_p": 0.9,
        "description": "Small GPT-2 model, fast and lightweight"
    },
    "gpt2-medium": {
        "name": "gpt2-medium", 
        "max_length": 512,
        "temperature": 0.8,
        "top_p": 0.85,
        "description": "Medium GPT-2 model, balanced performance"
    },
    "gpt2-large": {
        "name": "gpt2-large",
        "max_length": 1024, 
        "temperature": 0.6,
        "top_p": 0.95,
        "description": "Large GPT-2 model, best quality"
    },
    "dialogpt": {
        "name": "microsoft/DialoGPT-medium",
        "max_length": 512,
        "temperature": 0.7,
        "top_p": 0.9,
        "description": "Conversational model for dialogue"
    },
    "t5": {
        "name": "t5-base",
        "max_length": 256,
        "temperature": 0.7,
        "top_p": 0.9,
        "description": "Text-to-text transfer transformer"
    }
}

DART_PRESETS = {
    "fast": {
        "batch_size": 64,
        "epochs": 5,
        "learning_rate": 1e-4,
        "noise_std": 0.05
    },
    "balanced": {
        "batch_size": 32,
        "epochs": 10,
        "learning_rate": 1e-5,
        "noise_std": 0.1
    },
    "thorough": {
        "batch_size": 16,
        "epochs": 20,
        "learning_rate": 5e-6,
        "noise_std": 0.15
    }
}

class DARTExperiment:
    """Simplified DART experiment runner."""
    
    def __init__(self, model_key: str, dart_preset: str = "balanced", **overrides):
        self.logger = logging.getLogger(__name__)
        
        if model_key not in MODELS:
            raise ValueError(f"Model '{model_key}' not found. Available: {list(MODELS.keys())}")
        
        self.model_config = MODELS[model_key].copy()
        self.dart_config = DART_PRESETS[dart_preset].copy()
        
        # Apply overrides
        for key, value in overrides.items():
            if key in ['temperature', 'max_length', 'top_p']:
                self.model_config[key] = value
            elif key in ['batch_size', 'epochs', 'learning_rate', 'noise_std']:
                self.dart_config[key] = value
        
        self.logger.info(f"Initialized experiment with {model_key} model and {dart_preset} preset")
    
    def show_config(self):
        """Display current configuration."""
        print(f"\n=== DART Experiment Configuration ===")
        print(f"Model: {self.model_config['name']}")
        print(f"Description: {self.model_config.get('description', 'N/A')}")
        print(f"Max Length: {self.model_config['max_length']}")
        print(f"Temperature: {self.model_config['temperature']}")
        print(f"Top-p: {self.model_config['top_p']}")
        print(f"\nDART Parameters:")
        print(f"Batch Size: {self.dart_config['batch_size']}")
        print(f"Epochs: {self.dart_config['epochs']}")
        print(f"Learning Rate: {self.dart_config['learning_rate']}")
        print(f"Noise Std: {self.dart_config['noise_std']}")
        print("=" * 40)
    
    def run(self, simulate: bool = True):
        """Run the experiment."""
        self.show_config()
        
        if simulate:
            self._run_simulation()
        else:
            self.logger.warning("Full training not implemented - running simulation")
            self._run_simulation()
    
    def _run_simulation(self):
        """Run experiment simulation."""
        self.logger.info("Starting DART simulation...")
        
        epochs = self.dart_config['epochs']
        for epoch in range(epochs):
            # Simulate training metrics
            loss = 1.0 / (epoch + 1)
            reward = epoch * 0.1
            noise_std = self.dart_config['noise_std'] * (0.9 ** epoch)
            
            self.logger.info(f"Epoch {epoch+1}/{epochs}: loss={loss:.4f}, reward={reward:.4f}, noise_std={noise_std:.4f}")
            
            # Simulate evaluation
            if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
                eval_score = 0.7 + epoch * 0.01
                self.logger.info(f"Evaluation at epoch {epoch+1}: score={eval_score:.4f}")
        
        self.logger.info("Simulation completed!")

def list_models():
    """List available models."""
    print("\nAvailable Models:")
    print("-" * 50)
    for key, config in MODELS.items():
        print(f"{key:12} - {config['description']}")
    print()

def list_presets():
    """List available DART presets."""
    print("\nAvailable DART Presets:")
    print("-" * 50)
    for preset, config in DART_PRESETS.items():
        print(f"{preset:10} - epochs: {config['epochs']}, batch: {config['batch_size']}, lr: {config['learning_rate']}")
    print()

def save_config(model_key: str, dart_preset: str, output_path: str, **overrides):
    """Save configuration to simple text file."""
    model_config = MODELS[model_key].copy()
    dart_config = DART_PRESETS[dart_preset].copy()
    
    # Apply overrides
    for key, value in overrides.items():
        if key in ['temperature', 'max_length', 'top_p']:
            model_config[key] = value
        elif key in ['batch_size', 'epochs', 'learning_rate', 'noise_std']:
            dart_config[key] = value
    
    config_text = f"""# DART Configuration: {model_key} with {dart_preset} preset

[Model Configuration]
name = {model_config['name']}
max_length = {model_config['max_length']}
temperature = {model_config['temperature']}
top_p = {model_config['top_p']}
description = {model_config['description']}

[DART Parameters]
batch_size = {dart_config['batch_size']}
epochs = {dart_config['epochs']}
learning_rate = {dart_config['learning_rate']}
noise_std = {dart_config['noise_std']}

[Overrides Applied]
{chr(10).join(f"{k} = {v}" for k, v in overrides.items()) if overrides else "None"}
"""
    
    with open(output_path, 'w') as f:
        f.write(config_text)
    
    print(f"Configuration saved to: {output_path}")

def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="DART - Simplified LLM Testing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  dart-test --list-models                    # List available models
  dart-test --model gpt2                     # Quick test with GPT-2
  dart-test --model gpt2 --preset fast       # Fast training preset
  dart-test --model gpt2 --temperature 0.9   # Custom temperature
  dart-test --model dialogpt --epochs 15     # Custom epochs
        """
    )
    
    # Info commands
    parser.add_argument('--list-models', action='store_true', help='List available models')
    parser.add_argument('--list-presets', action='store_true', help='List DART presets')
    
    # Main options
    parser.add_argument('--model', type=str, choices=list(MODELS.keys()), 
                       help='Model to use for testing')
    parser.add_argument('--preset', type=str, choices=list(DART_PRESETS.keys()), 
                       default='balanced', help='DART preset (default: balanced)')
    
    # Model parameters
    parser.add_argument('--temperature', type=float, help='Sampling temperature')
    parser.add_argument('--max-length', type=int, help='Maximum sequence length')
    parser.add_argument('--top-p', type=float, help='Top-p sampling parameter')
    
    # DART parameters  
    parser.add_argument('--batch-size', type=int, help='Training batch size')
    parser.add_argument('--epochs', type=int, help='Number of training epochs')
    parser.add_argument('--learning-rate', type=float, help='Learning rate')
    parser.add_argument('--noise-std', type=float, help='Noise standard deviation')
    
    # Actions
    parser.add_argument('--run', action='store_true', help='Run the experiment')
    parser.add_argument('--save-config', type=str, help='Save config to file')
    parser.add_argument('--log-level', type=str, default='INFO', 
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Handle info commands
    if args.list_models:
        list_models()
        return
    
    if args.list_presets:
        list_presets()
        return
    
    # Require model for other operations
    if not args.model:
        parser.print_help()
        print("\nError: --model is required for experiments")
        sys.exit(1)
    
    # Collect overrides
    overrides = {}
    if args.temperature is not None:
        overrides['temperature'] = args.temperature
    if args.max_length is not None:
        overrides['max_length'] = args.max_length
    if args.top_p is not None:
        overrides['top_p'] = args.top_p
    if args.batch_size is not None:
        overrides['batch_size'] = args.batch_size
    if args.epochs is not None:
        overrides['epochs'] = args.epochs
    if args.learning_rate is not None:
        overrides['learning_rate'] = args.learning_rate
    if args.noise_std is not None:
        overrides['noise_std'] = args.noise_std
    
    # Save config if requested
    if args.save_config:
        save_config(args.model, args.preset, args.save_config, **overrides)
    
    # Create and run experiment
    try:
        experiment = DARTExperiment(args.model, args.preset, **overrides)
        
        if args.run or not args.save_config:
            experiment.run(simulate=True)
        
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()