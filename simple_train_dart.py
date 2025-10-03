#!/usr/bin/env python3
"""
Simple DART Training Script
Clean implementation of Algorithm 1 from the DART paper

Algorithm 1: DART Training (6 steps)
For each batch:
    1. embeddings = embedder(prompts)
    2. noise = noise_scheduler.sample()
    3. modified_prompts = vec2text(emb - noise)
    4. rewards = reward_model(emb, modified_emb)
    5. loss = ppo_loss(log_probs, rewards)
    6. optimizer.step()

Usage:
    uv run python simple_train_dart.py --dataset problem.csv --epochs 10
"""

import torch
import argparse
import logging
from tqdm import tqdm

from dart_system.training import (
    DARTDataset, DARTDatasetConfig,
    DiffusionNoiseScheduler, NoiseSchedulerConfig,
    Vec2TextWrapper,
    PPOLoss, PPOLossConfig, RewardModel
)
from dart_system.embedding.chinese_embedding import ChineseEmbeddingModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train_dart(dataset_path: str, num_epochs: int = 10, batch_size: int = 32,
               learning_rate: float = 1e-4, device: str = "cuda"):
    """Main DART training function"""

    # Setup device
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    logger.info(f"Training on: {device}")

    # Initialize components
    logger.info("Initializing components...")

    # Dataset
    dataset_config = DARTDatasetConfig(csv_path=dataset_path, batch_size=batch_size)
    dataset = DARTDataset(dataset_config)
    dataset.load_reference_prompts()

    # Embedder: P → ℝ^d
    embedder = ChineseEmbeddingModel(
        model_name="uer/sbert-base-chinese-nli",
        device=str(device)
    )

    # Noise scheduler: σ annealing
    noise_config = NoiseSchedulerConfig(
        initial_sigma=1.0,
        final_sigma=0.01,
        anneal_strategy="cosine",
        device=str(device)
    )
    noise_scheduler = DiffusionNoiseScheduler(noise_config)

    # Vec2text: e → P
    vec2text = Vec2TextWrapper(device=str(device))

    # Reward model: r(P, M_s(P_mod))
    reward_model = RewardModel(embedding_dim=768).to(device)

    # PPO Loss
    loss_config = PPOLossConfig(beta=0.01, clip_epsilon=0.2, device=str(device))
    ppo_loss = PPOLoss(loss_config)

    # Optimizer
    optimizer = torch.optim.Adam(reward_model.parameters(), lr=learning_rate)

    logger.info("Training started!")

    # Training loop
    for epoch in range(num_epochs):
        reward_model.train()
        epoch_loss = 0.0
        num_batches = 0

        # Update noise level
        noise_scheduler.step(num_epochs, epoch)
        sigma = noise_scheduler.get_sigma()

        # Iterate over batches
        for batch_prompts in tqdm(dataset, desc=f"Epoch {epoch+1}/{num_epochs}"):

            # Step 1: embeddings = embedder(prompts)
            embeddings = embedder.embed_texts(batch_prompts)

            # Step 2: noise = noise_scheduler.sample()
            noise = noise_scheduler.sample_noise(embeddings)

            # Step 3: modified_prompts = vec2text(emb - noise)
            modified_prompts = vec2text(embeddings, noise, batch_prompts)

            # Step 4: rewards = reward_model(emb, modified_emb)
            modified_embeddings = embedder.embed_texts(modified_prompts)
            rewards = reward_model(embeddings, modified_embeddings)

            # Step 5: loss = ppo_loss(log_probs, rewards)
            # Generate dummy policy outputs for demonstration
            log_probs = torch.randn(len(batch_prompts), device=device)
            old_log_probs = log_probs.detach()
            advantages = rewards - rewards.mean()

            loss, stats = ppo_loss(
                log_probs=log_probs,
                old_log_probs=old_log_probs,
                advantages=advantages,
                rewards=rewards,
                model_params=embeddings,
                target_params=modified_embeddings
            )

            # Step 6: optimizer.step()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(reward_model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        # Epoch summary
        avg_loss = epoch_loss / max(num_batches, 1)
        logger.info(f"Epoch {epoch+1} | Loss: {avg_loss:.4f} | σ: {sigma:.4f}")

    logger.info("Training complete!")

    # Save model
    torch.save(reward_model.state_dict(), "dart_reward_model.pt")
    logger.info("Model saved to: dart_reward_model.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simple DART Training")
    parser.add_argument("--dataset", type=str, required=True, help="Path to CSV dataset")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")

    args = parser.parse_args()

    train_dart(
        dataset_path=args.dataset,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        device=args.device
    )
