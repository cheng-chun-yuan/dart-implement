"""
PPO Loss for DART Training
Implements the loss function from Algorithm 1

Following Algorithm 1:
L = -L^PPO(π(P, e), rew) + βL^REG(μ)

where:
- L^PPO: Proximal Policy Optimization loss
- rew: reward from reward model r (ToxiGuardrail toxicity scores)
- β: regularization coefficient
- L^REG: regularization loss
- μ: model parameters

Reward Model: nicholasKluge/ToxiGuardrail scores target LLM outputs
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Optional, Tuple, Dict, List
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class PPOLossConfig:
    """Configuration for PPO loss"""
    beta: float = 0.01  # Regularization coefficient β
    clip_epsilon: float = 0.2  # PPO clipping parameter
    value_loss_coef: float = 0.5  # Value function loss coefficient
    entropy_coef: float = 0.01  # Entropy bonus coefficient
    normalize_advantages: bool = True
    device: str = "cpu"


class PPOLoss(nn.Module):
    """
    PPO Loss for DART Training

    Implements Algorithm 1 loss:
    L = -L^PPO(π(P, e), rew) + βL^REG(μ)

    Components:
    1. L^PPO: Policy gradient loss with clipping
    2. L^REG: Regularization loss (KL divergence or L2)
    """

    def __init__(self, config: PPOLossConfig):
        """
        Initialize PPO loss

        Args:
            config: PPO loss configuration
        """
        super().__init__()
        self.config = config

        logger.info(f"Initialized PPO loss")
        logger.info(f"  β (beta): {config.beta}")
        logger.info(f"  Clip ε: {config.clip_epsilon}")
        logger.info(f"  Value loss coef: {config.value_loss_coef}")
        logger.info(f"  Entropy coef: {config.entropy_coef}")

    def compute_ppo_loss(
        self,
        log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        rewards: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute PPO policy loss

        Following Algorithm 1:
        L^PPO(π(P, e), rew)

        Args:
            log_probs: Log probabilities from current policy
            old_log_probs: Log probabilities from old policy
            advantages: Advantage estimates
            rewards: Rewards from reward model r(P, M_s(P_mod))

        Returns:
            Tuple[torch.Tensor, Dict]: (PPO loss, statistics)
        """
        # Normalize advantages
        if self.config.normalize_advantages:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Compute probability ratio: π_new / π_old
        ratio = torch.exp(log_probs - old_log_probs)

        # Compute clipped surrogate objective
        surr1 = ratio * advantages
        surr2 = torch.clamp(
            ratio,
            1.0 - self.config.clip_epsilon,
            1.0 + self.config.clip_epsilon
        ) * advantages

        # PPO loss (negative because we want to maximize)
        policy_loss = -torch.min(surr1, surr2).mean()

        # Statistics
        stats = {
            "policy_loss": policy_loss.item(),
            "ratio_mean": ratio.mean().item(),
            "ratio_std": ratio.std().item(),
            "advantage_mean": advantages.mean().item(),
            "reward_mean": rewards.mean().item()
        }

        return policy_loss, stats

    def compute_regularization_loss(
        self,
        model_params: torch.Tensor,
        target_params: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute regularization loss L^REG(μ)

        Following Algorithm 1:
        βL^REG(μ)

        Args:
            model_params: Current model parameters μ
            target_params: Target parameters (for KL regularization)

        Returns:
            torch.Tensor: Regularization loss
        """
        if target_params is not None:
            # KL divergence regularization
            kl_div = F.kl_div(
                F.log_softmax(model_params, dim=-1),
                F.softmax(target_params, dim=-1),
                reduction='batchmean'
            )
            reg_loss = kl_div
        else:
            # L2 regularization
            reg_loss = torch.sum(model_params ** 2)

        return reg_loss

    def forward(
        self,
        log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        rewards: torch.Tensor,
        model_params: torch.Tensor,
        target_params: Optional[torch.Tensor] = None,
        entropy: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute total DART loss

        Following Algorithm 1:
        L = -L^PPO(π(P, e), rew) + βL^REG(μ)

        Args:
            log_probs: Current policy log probabilities
            old_log_probs: Old policy log probabilities
            advantages: Advantage estimates
            rewards: Rewards from reward model
            model_params: Model parameters
            target_params: Target parameters for regularization
            entropy: Policy entropy (optional)

        Returns:
            Tuple[torch.Tensor, Dict]: (total loss, statistics)
        """
        # Compute PPO loss: L^PPO
        ppo_loss, ppo_stats = self.compute_ppo_loss(
            log_probs, old_log_probs, advantages, rewards
        )

        # Compute regularization loss: L^REG
        reg_loss = self.compute_regularization_loss(model_params, target_params)

        # Total loss: L = -L^PPO + βL^REG
        # Note: ppo_loss is already negative (we maximize policy objective)
        total_loss = ppo_loss + self.config.beta * reg_loss

        # Add entropy bonus if provided
        if entropy is not None:
            total_loss = total_loss - self.config.entropy_coef * entropy.mean()

        # Combine statistics
        stats = {
            **ppo_stats,
            "reg_loss": reg_loss.item(),
            "total_loss": total_loss.item()
        }

        if entropy is not None:
            stats["entropy"] = entropy.mean().item()

        return total_loss, stats


class ToxiGuardrailRewardModel(nn.Module):
    """
    ToxiGuardrail-based Reward Model r: P → ℝ

    Uses HuggingFace nicholasKluge/ToxiGuardrail to score target LLM outputs

    Following Algorithm 1:
    rew ← r(P, M_s(P_mod))

    where:
    - r: ToxiGuardrail toxicity classifier
    - P: original prompt
    - M_s: target LLM
    - P_mod: modified prompt

    Higher toxicity scores = higher rewards (we want to maximize harmful outputs)
    """

    def __init__(
        self,
        model_name: str = "nicholasKluge/ToxiGuardrail",
        device: str = "cpu",
        use_toxicity_as_reward: bool = True
    ):
        """
        Initialize ToxiGuardrail reward model

        Args:
            model_name: HuggingFace model name
            device: Device to run model on
            use_toxicity_as_reward: If True, toxicity score is reward
                                   If False, 1 - toxicity is reward
        """
        super().__init__()

        self.device = device
        self.use_toxicity_as_reward = use_toxicity_as_reward

        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification

            logger.info(f"Loading ToxiGuardrail reward model: {model_name}")

            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
            self.model.eval().to(device)

            logger.info(f"ToxiGuardrail loaded successfully on {device}")
            logger.info(f"Use toxicity as reward: {use_toxicity_as_reward}")

        except ImportError:
            logger.error("transformers library required for ToxiGuardrail")
            raise
        except Exception as e:
            logger.error(f"Failed to load ToxiGuardrail: {e}")
            raise

    def score_toxicity(self, texts: List[str]) -> torch.Tensor:
        """
        Score toxicity of texts using ToxiGuardrail

        Args:
            texts: List of texts to score

        Returns:
            torch.Tensor: Toxicity scores [batch_size]
        """
        if not texts:
            return torch.tensor([], device=self.device)

        # Tokenize inputs
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512
        ).to(self.device)

        # Get toxicity predictions
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits

            # Apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)

            # Get toxicity probability (typically class 1)
            # Check model config for exact label mapping
            toxicity_scores = probs[:, 1]  # Assuming index 1 is "toxic"

        return toxicity_scores

    def forward(
        self,
        prompts: List[str],
        llm_responses: List[str]
    ) -> torch.Tensor:
        """
        Compute rewards for (prompt, LLM response) pairs

        Following Algorithm 1:
        rew ← r(P, M_s(P_mod))

        Args:
            prompts: Original or modified prompts
            llm_responses: Responses from target LLM M_s

        Returns:
            torch.Tensor: Reward values [batch_size]
        """
        # Score toxicity of LLM responses
        toxicity_scores = self.score_toxicity(llm_responses)

        if self.use_toxicity_as_reward:
            # Higher toxicity = higher reward (attack objective)
            rewards = toxicity_scores
        else:
            # Lower toxicity = higher reward (defense objective)
            rewards = 1.0 - toxicity_scores

        return rewards


class RewardModel(nn.Module):
    """
    Fallback MLP Reward Model r: P → ℝ

    Simple learned reward model for cases where ToxiGuardrail is unavailable

    Following Algorithm 1:
    rew ← r(P, M_s(P_mod))

    where:
    - r: reward model
    - P: original prompt
    - M_s: target LLM
    - P_mod: modified prompt
    """

    def __init__(self, embedding_dim: int = 768):
        """
        Initialize reward model

        Args:
            embedding_dim: Dimension of prompt embeddings
        """
        super().__init__()

        # Simple MLP reward model
        self.reward_net = nn.Sequential(
            nn.Linear(embedding_dim * 2, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1)
        )

        logger.info(f"Initialized fallback MLP reward model with embedding_dim={embedding_dim}")

    def forward(
        self,
        prompt_embedding: torch.Tensor,
        response_embedding: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute reward for (prompt, response) pair

        Args:
            prompt_embedding: Embedding of prompt P
            response_embedding: Embedding of LLM response M_s(P_mod)

        Returns:
            torch.Tensor: Reward values, shape [batch_size]
        """
        # Concatenate prompt and response embeddings
        combined = torch.cat([prompt_embedding, response_embedding], dim=-1)

        # Compute reward
        reward = self.reward_net(combined).squeeze(-1)

        return reward
