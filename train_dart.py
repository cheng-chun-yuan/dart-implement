#!/usr/bin/env python3
"""
DART Training Script
Main entry point for training DART adversarial text generation

Usage:
    uv run python train_dart.py --dataset problem.csv --epochs 10
"""

import argparse
import logging
import sys
from pathlib import Path

from dart_system.training.dart_trainer import DARTTrainer, DARTTrainerConfig


def setup_logging(log_file: str = None):
    """Setup logging configuration"""
    handlers = [logging.StreamHandler(sys.stdout)]

    if log_file:
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Train DART adversarial text generation system'
    )

    # Dataset
    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        help='Path to CSV dataset with reference prompts'
    )

    # Training parameters
    parser.add_argument(
        '--epochs',
        type=int,
        default=10,
        help='Number of training epochs (default: 10)'
    )

    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Training batch size (default: 32)'
    )

    parser.add_argument(
        '--lr',
        type=float,
        default=1e-4,
        help='Learning rate (default: 1e-4)'
    )

    # Model parameters
    parser.add_argument(
        '--embedding-dim',
        type=int,
        default=768,
        help='Embedding dimension (default: 768)'
    )

    parser.add_argument(
        '--beta',
        type=float,
        default=0.01,
        help='Regularization coefficient β (default: 0.01)'
    )

    # Noise scheduling
    parser.add_argument(
        '--initial-sigma',
        type=float,
        default=1.0,
        help='Initial noise level σ (default: 1.0)'
    )

    parser.add_argument(
        '--final-sigma',
        type=float,
        default=0.01,
        help='Final noise level σ (default: 0.01)'
    )

    parser.add_argument(
        '--anneal-strategy',
        type=str,
        default='cosine',
        choices=['linear', 'cosine', 'exponential'],
        help='Sigma annealing strategy (default: cosine)'
    )

    # Device
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        choices=['cuda', 'cpu'],
        help='Device to use for training (default: cuda)'
    )

    # Checkpointing
    parser.add_argument(
        '--checkpoint-dir',
        type=str,
        default='checkpoints',
        help='Directory to save checkpoints (default: checkpoints)'
    )

    parser.add_argument(
        '--save-interval',
        type=int,
        default=100,
        help='Save checkpoint every N steps (default: 100)'
    )

    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to checkpoint to resume from'
    )

    # Logging
    parser.add_argument(
        '--log-file',
        type=str,
        default=None,
        help='Path to log file (default: None, stdout only)'
    )

    parser.add_argument(
        '--log-interval',
        type=int,
        default=10,
        help='Log every N steps (default: 10)'
    )

    return parser.parse_args()


def main():
    """Main training function"""
    args = parse_args()

    # Setup logging
    setup_logging(args.log_file)
    logger = logging.getLogger(__name__)

    logger.info("=" * 80)
    logger.info("DART Training Script")
    logger.info("=" * 80)

    # Validate dataset path
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        logger.error(f"Dataset file not found: {dataset_path}")
        sys.exit(1)

    # Create checkpoint directory
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Create trainer configuration
    config = DARTTrainerConfig(
        csv_path=str(dataset_path),
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        embedding_dim=args.embedding_dim,
        beta=args.beta,
        initial_sigma=args.initial_sigma,
        final_sigma=args.final_sigma,
        anneal_strategy=args.anneal_strategy,
        device=args.device,
        checkpoint_dir=str(checkpoint_dir),
        save_interval=args.save_interval,
        log_interval=args.log_interval
    )

    # Log configuration
    logger.info("\nTraining Configuration:")
    logger.info(f"  Dataset: {dataset_path}")
    logger.info(f"  Epochs: {config.num_epochs}")
    logger.info(f"  Batch size: {config.batch_size}")
    logger.info(f"  Learning rate: {config.learning_rate}")
    logger.info(f"  Embedding dim: {config.embedding_dim}")
    logger.info(f"  β (regularization): {config.beta}")
    logger.info(f"  Initial σ: {config.initial_sigma}")
    logger.info(f"  Final σ: {config.final_sigma}")
    logger.info(f"  Anneal strategy: {config.anneal_strategy}")
    logger.info(f"  Device: {config.device}")
    logger.info(f"  Checkpoint dir: {checkpoint_dir}")
    logger.info("")

    # Initialize trainer
    logger.info("Initializing DART trainer...")
    trainer = DARTTrainer(config)

    # Resume from checkpoint if specified
    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)

    # Train
    logger.info("\nStarting training...")
    try:
        training_stats = trainer.train()

        # Save final checkpoint
        final_checkpoint = checkpoint_dir / "final_checkpoint.pt"
        trainer.save_checkpoint(str(final_checkpoint))

        # Print final statistics
        logger.info("\n" + "=" * 80)
        logger.info("Training Complete!")
        logger.info("=" * 80)
        logger.info("\nFinal Statistics:")

        final_stats = training_stats[-1]
        for key, value in final_stats.items():
            logger.info(f"  {key}: {value:.4f}")

        logger.info(f"\nFinal checkpoint saved to: {final_checkpoint}")
        logger.info("=" * 80)

    except KeyboardInterrupt:
        logger.info("\n\nTraining interrupted by user")

        # Save interrupt checkpoint
        interrupt_checkpoint = checkpoint_dir / "interrupt_checkpoint.pt"
        trainer.save_checkpoint(str(interrupt_checkpoint))
        logger.info(f"Checkpoint saved to: {interrupt_checkpoint}")

    except Exception as e:
        logger.error(f"\n\nTraining failed with error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
