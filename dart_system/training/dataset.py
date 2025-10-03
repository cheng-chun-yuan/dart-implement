"""
DART Training Dataset
Implements reference prompt dataset loading following Algorithm 1
"""

import torch
import random
import logging
from typing import List, Tuple, Dict, Optional, Iterator
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class DARTDatasetConfig:
    """Configuration for DART training dataset"""
    csv_path: str
    batch_size: int = 32
    max_length: int = 128
    shuffle: bool = True
    random_seed: int = 42


class DARTDataset:
    """
    DART Training Dataset

    Manages reference prompts P (dataset of reference prompts)
    following Algorithm 1 specification.

    Core functionality:
    - Load reference prompts from CSV
    - Generate training batches
    - Support for both harmful and benign prompts
    """

    def __init__(self, config: DARTDatasetConfig):
        """
        Initialize DART dataset

        Args:
            config: Dataset configuration
        """
        self.config = config
        self.reference_prompts: List[str] = []
        self._current_epoch = 0

        # Set random seed for reproducibility
        random.seed(config.random_seed)
        torch.manual_seed(config.random_seed)

        logger.info(f"Initialized DART dataset with batch_size={config.batch_size}")

    def load_reference_prompts(self) -> List[str]:
        """
        Load dataset of reference prompts P from CSV

        Following Algorithm 1:
        dataset of reference prompts P

        Returns:
            List[str]: Reference prompts
        """
        import csv

        logger.info(f"Loading reference prompts from: {self.config.csv_path}")

        prompts = []
        with open(self.config.csv_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader)  # Skip header

            for row in reader:
                if row and row[0].strip():
                    prompt = row[0].strip()
                    # Truncate to max_length
                    if len(prompt) > self.config.max_length:
                        prompt = prompt[:self.config.max_length]
                    prompts.append(prompt)

        self.reference_prompts = prompts
        logger.info(f"Loaded {len(self.reference_prompts)} reference prompts")

        return self.reference_prompts

    def get_batches(self) -> Iterator[List[str]]:
        """
        Generate training batches following Algorithm 1

        Algorithm 1 structure:
        for i ≤ num_epochs do
            for P ∈ P do
                # Process each prompt in batch

        Yields:
            List[str]: Batch of prompts
        """
        prompts = self.reference_prompts.copy()

        if self.config.shuffle:
            random.shuffle(prompts)

        # Generate batches
        for i in range(0, len(prompts), self.config.batch_size):
            batch = prompts[i:i + self.config.batch_size]
            yield batch

    def __len__(self) -> int:
        """Get dataset size"""
        return len(self.reference_prompts)

    def __iter__(self) -> Iterator[List[str]]:
        """Iterate over batches"""
        return self.get_batches()

    def get_statistics(self) -> Dict[str, float]:
        """
        Get dataset statistics

        Returns:
            Dict: Statistics including size, avg length, etc.
        """
        if not self.reference_prompts:
            return {}

        lengths = [len(p) for p in self.reference_prompts]

        return {
            "total_prompts": len(self.reference_prompts),
            "avg_length": sum(lengths) / len(lengths),
            "min_length": min(lengths),
            "max_length": max(lengths),
            "batches_per_epoch": (len(self.reference_prompts) + self.config.batch_size - 1) // self.config.batch_size
        }
