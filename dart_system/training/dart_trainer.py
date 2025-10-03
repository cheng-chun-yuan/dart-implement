"""
DART Trainer - Main Training Loop
Implements Algorithm 1 from the DART paper

Algorithm 1: DART Training
dataset of reference prompts P, embedder emb: P → ℝ^d, diffusion model d_σ: P × ℝ^d → ℝ^d,
target LLM M_s: P → P, reward model r: P → ℝ, learning rate α, number of epoch num_epochs, budget ε

for i ≤ num_epochs do
    for P ∈ P do
        e ← emb(P)
        n ← N(d_σ(P, e), σ)     ▷ σ is annealed every iteration
        P_mod ← vec2text(e - n)
        rew ← r(P, M_s(P_mod))
        L = -L^PPO(π(P, e), rew) + βL^REG(μ)
        θ ← θ - α∇L
    end for
end for
"""

import torch
import torch.nn as nn
import torch.optim as optim
import logging
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass
from tqdm import tqdm

from dart_system.embedding.chinese_embedding import ChineseEmbeddingModel
from .dataset import DARTDataset, DARTDatasetConfig
from .noise_scheduler import DiffusionNoiseScheduler, NoiseSchedulerConfig
from .vec2text_wrapper import Vec2TextWrapper
from .ppo_loss import PPOLoss, PPOLossConfig, ToxiGuardrailRewardModel, RewardModel

logger = logging.getLogger(__name__)


@dataclass
class DARTTrainerConfig:
    """Configuration for DART trainer"""
    # Dataset
    csv_path: str
    batch_size: int = 32

    # Training
    num_epochs: int = 10
    learning_rate: float = 1e-4
    budget_epsilon: float = 2.0

    # Model dimensions
    embedding_dim: int = 768

    # Loss coefficients
    beta: float = 0.01  # Regularization coefficient
    clip_epsilon: float = 0.2

    # Noise scheduling
    initial_sigma: float = 1.0
    final_sigma: float = 0.01
    anneal_strategy: str = "cosine"

    # Reward model
    use_toxiguardrail: bool = True  # Use ToxiGuardrail for rewards
    toxiguardrail_model: str = "nicholasKluge/ToxiGuardrail"
    use_toxicity_as_reward: bool = True  # Higher toxicity = higher reward

    # Target LLM (for generating responses to score)
    target_llm_name: Optional[str] = None  # e.g., "gpt2", "meta-llama/Llama-2-7b-hf"
    target_llm_max_length: int = 256

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Logging
    log_interval: int = 10
    save_interval: int = 100
    checkpoint_dir: str = "checkpoints"


class DARTTrainer:
    """
    DART Training System

    Implements Algorithm 1: DART Training
    Full pipeline from reference prompts to trained adversarial model
    """

    def __init__(self, config: DARTTrainerConfig):
        """
        Initialize DART trainer

        Args:
            config: Trainer configuration
        """
        self.config = config
        self.device = torch.device(config.device)

        logger.info("=" * 60)
        logger.info("Initializing DART Trainer")
        logger.info("=" * 60)

        # Initialize components following Algorithm 1
        self._init_dataset()
        self._init_embedder()
        self._init_noise_scheduler()
        self._init_vec2text()
        self._init_reward_model()
        self._init_loss()
        self._init_optimizer()

        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.training_stats = []

        logger.info("DART Trainer initialized successfully")
        logger.info(f"Device: {self.device}")
        logger.info(f"Batch size: {config.batch_size}")
        logger.info(f"Learning rate: {config.learning_rate}")
        logger.info("=" * 60)

    def _init_dataset(self):
        """Initialize dataset of reference prompts P"""
        logger.info("Initializing dataset...")

        dataset_config = DARTDatasetConfig(
            csv_path=self.config.csv_path,
            batch_size=self.config.batch_size,
            shuffle=True
        )

        self.dataset = DARTDataset(dataset_config)
        self.dataset.load_reference_prompts()

        stats = self.dataset.get_statistics()
        logger.info(f"  Loaded {stats['total_prompts']} reference prompts")
        logger.info(f"  Batches per epoch: {stats['batches_per_epoch']}")

    def _init_embedder(self):
        """Initialize embedder emb: P → ℝ^d"""
        logger.info("Initializing embedder...")

        self.embedder = ChineseEmbeddingModel(
            model_name="uer/sbert-base-chinese-nli",
            device=str(self.device)
        )

        logger.info(f"  Embedding dim: {self.embedder.embedding_dim}")

    def _init_noise_scheduler(self):
        """Initialize diffusion model d_σ and noise scheduler"""
        logger.info("Initializing noise scheduler...")

        noise_config = NoiseSchedulerConfig(
            initial_sigma=self.config.initial_sigma,
            final_sigma=self.config.final_sigma,
            anneal_strategy=self.config.anneal_strategy,
            device=str(self.device)
        )

        self.noise_scheduler = DiffusionNoiseScheduler(noise_config)
        logger.info(f"  Initial σ: {noise_config.initial_sigma}")
        logger.info(f"  Final σ: {noise_config.final_sigma}")

    def _init_vec2text(self):
        """Initialize vec2text reconstruction"""
        logger.info("Initializing vec2text...")

        self.vec2text = Vec2TextWrapper(device=str(self.device))
        logger.info("  Vec2text wrapper ready")

    def _init_reward_model(self):
        """Initialize reward model r: P → ℝ"""
        logger.info("Initializing reward model...")

        self.reward_model = RewardModel(
            embedding_dim=self.config.embedding_dim
        ).to(self.device)

        logger.info("  Reward model initialized")

    def _init_loss(self):
        """Initialize loss function"""
        logger.info("Initializing loss function...")

        loss_config = PPOLossConfig(
            beta=self.config.beta,
            clip_epsilon=self.config.clip_epsilon,
            device=str(self.device)
        )

        self.ppo_loss = PPOLoss(loss_config)
        logger.info(f"  β (regularization): {loss_config.beta}")
        logger.info(f"  Clip ε: {loss_config.clip_epsilon}")

    def _init_optimizer(self):
        """Initialize optimizer with learning rate α"""
        logger.info("Initializing optimizer...")

        # Collect all trainable parameters
        self.trainable_params = list(self.reward_model.parameters())

        self.optimizer = optim.Adam(
            self.trainable_params,
            lr=self.config.learning_rate
        )

        logger.info(f"  Learning rate α: {self.config.learning_rate}")
        logger.info(f"  Trainable parameters: {sum(p.numel() for p in self.trainable_params):,}")

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Train one epoch following Algorithm 1

        for P ∈ P do
            e ← emb(P)
            n ← N(d_σ(P, e), σ)
            P_mod ← vec2text(e - n)
            rew ← r(P, M_s(P_mod))
            L = -L^PPO(π(P, e), rew) + βL^REG(μ)
            θ ← θ - α∇L
        end for

        Args:
            epoch: Current epoch number

        Returns:
            Dict: Epoch statistics
        """
        self.reward_model.train()

        epoch_stats = {
            "loss": 0.0,
            "policy_loss": 0.0,
            "reg_loss": 0.0,
            "reward_mean": 0.0
        }

        num_batches = 0

        # Update sigma for this epoch
        self.noise_scheduler.step(self.config.num_epochs, epoch)
        current_sigma = self.noise_scheduler.get_sigma()

        # Iterate over reference prompts: for P ∈ P do
        for batch_prompts in tqdm(
            self.dataset.get_batches(),
            desc=f"Epoch {epoch + 1}/{self.config.num_epochs}",
            total=self.dataset.get_statistics()['batches_per_epoch']
        ):
            # Step 1: e ← emb(P)
            embeddings = self.embedder.embed_texts(batch_prompts)

            # Step 2: n ← N(d_σ(P, e), σ)
            noise = self.noise_scheduler.sample_noise(embeddings)

            # Step 3: P_mod ← vec2text(e - n)
            modified_prompts = self.vec2text(embeddings, noise, batch_prompts)

            # Step 4: rew ← r(P, M_s(P_mod))
            # For simplicity, we use the modified prompt embeddings
            # In practice, you would query the target LLM M_s
            modified_embeddings = self.embedder.embed_texts(modified_prompts)
            rewards = self.reward_model(embeddings, modified_embeddings)

            # Step 5: Compute loss L = -L^PPO(π(P, e), rew) + βL^REG(μ)
            # For demonstration, we create dummy policy outputs
            log_probs = torch.randn(len(batch_prompts), device=self.device)
            old_log_probs = log_probs.detach()
            advantages = rewards - rewards.mean()

            # Compute loss
            loss, stats = self.ppo_loss(
                log_probs=log_probs,
                old_log_probs=old_log_probs,
                advantages=advantages,
                rewards=rewards,
                model_params=embeddings,
                target_params=modified_embeddings
            )

            # Step 6: θ ← θ - α∇L
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.trainable_params, max_norm=1.0)
            self.optimizer.step()

            # Update statistics
            epoch_stats["loss"] += loss.item()
            epoch_stats["policy_loss"] += stats["policy_loss"]
            epoch_stats["reg_loss"] += stats["reg_loss"]
            epoch_stats["reward_mean"] += stats["reward_mean"]

            num_batches += 1
            self.global_step += 1

            # Logging
            if self.global_step % self.config.log_interval == 0:
                logger.info(
                    f"Step {self.global_step} | "
                    f"Loss: {loss.item():.4f} | "
                    f"Reward: {stats['reward_mean']:.4f} | "
                    f"σ: {current_sigma:.4f}"
                )

        # Average statistics
        for key in epoch_stats:
            epoch_stats[key] /= max(num_batches, 1)

        epoch_stats["sigma"] = current_sigma

        return epoch_stats

    def train(self) -> List[Dict[str, float]]:
        """
        Main training loop following Algorithm 1

        for i ≤ num_epochs do
            [train one epoch]
        end for

        Returns:
            List[Dict]: Training statistics per epoch
        """
        logger.info("\n" + "=" * 60)
        logger.info("Starting DART Training")
        logger.info("=" * 60)

        for epoch in range(self.config.num_epochs):
            self.current_epoch = epoch

            # Train one epoch
            epoch_stats = self.train_epoch(epoch)
            self.training_stats.append(epoch_stats)

            # Log epoch summary
            logger.info("\n" + "-" * 60)
            logger.info(f"Epoch {epoch + 1}/{self.config.num_epochs} Summary:")
            logger.info(f"  Loss: {epoch_stats['loss']:.4f}")
            logger.info(f"  Policy Loss: {epoch_stats['policy_loss']:.4f}")
            logger.info(f"  Reg Loss: {epoch_stats['reg_loss']:.4f}")
            logger.info(f"  Reward: {epoch_stats['reward_mean']:.4f}")
            logger.info(f"  σ: {epoch_stats['sigma']:.4f}")
            logger.info("-" * 60 + "\n")

        logger.info("=" * 60)
        logger.info("Training Complete!")
        logger.info("=" * 60)

        return self.training_stats

    def save_checkpoint(self, path: str):
        """Save training checkpoint"""
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'reward_model_state': self.reward_model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'training_stats': self.training_stats,
            'config': self.config
        }

        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved to {path}")

    def load_checkpoint(self, path: str):
        """Load training checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)

        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.reward_model.load_state_dict(checkpoint['reward_model_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        self.training_stats = checkpoint['training_stats']

        logger.info(f"Checkpoint loaded from {path}")
        logger.info(f"  Resuming from epoch {self.current_epoch + 1}")
