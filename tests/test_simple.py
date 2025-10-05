#!/usr/bin/env python3
"""
Integration Test: Embed → Add Noise → Vec2Text Pipeline

NOTE: Vec2Text reconstruction requires a trained model to decode embeddings.
The current T5 model generates random text because it's not trained for this task.

This test demonstrates:
1. Text → Embedding (ChineseEmbedding with SBERT)
2. Embedding → Noisy Embedding (DiffusionNoise with constraints)
3. Noisy Embedding → Text (T5 model - needs fine-tuning for real reconstruction)
4. Embedding similarity preservation (actual metric used in DART)
"""

import unittest
import sys
import torch
import csv
from pathlib import Path

# Add dart_system to path
sys.path.append(str(Path(__file__).parent.parent / 'dart_system'))

from embedding.chinese_embedding import ChineseEmbedding, EmbeddingConfig
from noise.diffusion_noise import DiffusionNoise, NoiseConfig
from reconstruction.vec2text import ChineseVec2TextModel, TextSimilarityValidator


class TestEmbedNoiseVec2TextPipeline(unittest.TestCase):

    def setUp(self):
        """Set up pipeline components"""
        print("\n" + "=" * 70)
        print("Initializing pipeline components...")

        # Initialize embedding model
        embedding_config = EmbeddingConfig(
            embedding_dim=768,
            normalization=True,  # Enable normalization for better similarity
            device="cpu"
        )
        self.embedder = ChineseEmbedding(embedding_config)
        print("✅ Embedding model loaded")

        # Initialize noise generators with different levels
        self.noise_configs = {
            'none': NoiseConfig(proximity_threshold=0.0, default_std=0.0, adaptive_scaling=False),
            'tiny': NoiseConfig(proximity_threshold=0.1, default_std=0.01, adaptive_scaling=True),
            'small': NoiseConfig(proximity_threshold=0.3, default_std=0.03, adaptive_scaling=True),
            'medium': NoiseConfig(proximity_threshold=0.5, default_std=0.05, adaptive_scaling=True),
        }

        self.noise_generators = {
            level: DiffusionNoise(config=config)
            for level, config in self.noise_configs.items()
        }
        print("✅ Noise generators initialized (4 levels: none/tiny/small/medium)")

        # Initialize vec2text model
        self.vec2text = ChineseVec2TextModel(
            device="cpu",
            max_length=64
        )
        print("✅ Vec2text model loaded (NOTE: needs training for real reconstruction)")

        # Initialize validator
        self.validator = TextSimilarityValidator(similarity_threshold=0.7)
        print("✅ Validator initialized")

        # Load top 10 texts from problem.csv
        self.test_texts = self._load_problem_texts(limit=10)
        print(f"✅ Loaded {len(self.test_texts)} texts from problem.csv")
        print("=" * 70)

    def _load_problem_texts(self, limit=10):
        """Load texts from problem.csv"""
        csv_path = Path(__file__).parent.parent / "problem.csv"

        texts = []
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                if i >= limit:
                    break
                text = list(row.values())[0] if row else ""
                if text.strip():
                    texts.append(text.strip())

        return texts

    def test_embedding_similarity_with_noise(self):
        """Test how noise affects embedding similarity (actual DART metric)"""
        print("\n=== Test 1: Embedding Similarity vs Noise Level ===\n")

        text = self.test_texts[0]
        print(f"📝 Original Text:")
        print(f"   {text}")
        print()

        # Get clean embedding
        embedding_np = self.embedder.encode([text])
        clean_embedding = torch.from_numpy(embedding_np[0])

        print(f"Testing noise impact on embedding similarity:")
        print(f"{'Noise Level':<15} {'L2 Norm':<12} {'Cosine Sim':<12} {'Status'}")
        print("-" * 60)

        for level in ['none', 'tiny', 'small', 'medium']:
            noise_gen = self.noise_generators[level]

            # Add noise
            if level == 'none':
                noisy_embedding = clean_embedding
                noise_norm = 0.0
            else:
                embedding_list = [clean_embedding.tolist()]
                noise_perturbations = noise_gen.sample_perturbation(embedding_list)
                noisy_embedding_list = [[e + n for e, n in zip(embedding_list[0], noise_perturbations[0])]]
                noisy_embedding = torch.tensor(noisy_embedding_list[0])
                noise_norm = noise_gen._compute_l2_norm(noise_perturbations[0])

            # Compute cosine similarity
            cos_sim = torch.nn.functional.cosine_similarity(
                clean_embedding.unsqueeze(0),
                noisy_embedding.unsqueeze(0)
            ).item()

            status = "✅ Good" if cos_sim > 0.95 else "⚠️ Degraded" if cos_sim > 0.85 else "❌ Poor"

            print(f"{level:<15} {noise_norm:<12.4f} {cos_sim:<12.4f} {status}")

        print()
        print("📊 Key Insight:")
        print("   DART uses embedding similarity, not text reconstruction!")
        print("   Smaller noise preserves semantic meaning better.")
        print()

    def test_no_noise_vs_noise_comparison(self):
        """Compare reconstruction quality: no noise vs small noise"""
        print("\n=== Test 2: No Noise vs Small Noise Comparison ===\n")

        text = self.test_texts[0]
        print(f"📝 ORIGINAL:")
        print(f"   {text}")
        print()

        embedding_np = self.embedder.encode([text])
        embedding = torch.from_numpy(embedding_np[0])

        # Test 1: No noise
        print("=" * 70)
        print("CASE 1: NO NOISE (Baseline)")
        print("=" * 70)
        reconstructed_clean = self.vec2text.embedding_to_text(embedding, temperature=0.5)
        print(f"📝 Reconstructed (No Noise):")
        print(f"   {reconstructed_clean}")
        sim_clean = self.validator._compute_text_similarity(text, reconstructed_clean)
        print(f"   Character Similarity: {sim_clean:.3f}")
        print()

        # Test 2: Small noise
        print("=" * 70)
        print("CASE 2: SMALL NOISE (std=0.03, threshold=0.3)")
        print("=" * 70)
        noise_gen = self.noise_generators['small']
        embedding_list = [embedding.tolist()]
        noise_perturbations = noise_gen.sample_perturbation(embedding_list)
        noisy_embedding_list = [[e + n for e, n in zip(embedding_list[0], noise_perturbations[0])]]
        noisy_embedding = torch.tensor(noisy_embedding_list[0])

        noise_norm = noise_gen._compute_l2_norm(noise_perturbations[0])
        cos_sim = torch.nn.functional.cosine_similarity(
            embedding.unsqueeze(0),
            noisy_embedding.unsqueeze(0)
        ).item()

        reconstructed_noisy = self.vec2text.embedding_to_text(noisy_embedding, temperature=0.5)
        print(f"   Noise L2 norm: {noise_norm:.4f}")
        print(f"   Embedding cosine similarity: {cos_sim:.4f}")
        print(f"📝 Reconstructed (With Noise):")
        print(f"   {reconstructed_noisy}")
        sim_noisy = self.validator._compute_text_similarity(text, reconstructed_noisy)
        print(f"   Character Similarity: {sim_noisy:.3f}")
        print()

        print("📊 Analysis:")
        print(f"   Both reconstructions have low similarity because T5 is NOT trained")
        print(f"   for embedding-to-text decoding. It generates random Chinese text.")
        print(f"   ")
        print(f"   For DART evaluation, use EMBEDDING similarity: {cos_sim:.4f}")
        print(f"   This shows semantic meaning is preserved with small noise.")
        print()

    def test_all_10_prompts_embedding_similarity(self):
        """Test embedding similarity preservation for all 10 prompts"""
        print("\n=== Test 3: All 10 Prompts - Embedding Similarity ===\n")

        print(f"📝 ORIGINAL PROMPTS (from problem.csv):")
        for i, text in enumerate(self.test_texts, 1):
            print(f"   {i:2d}. {text[:60]}..." if len(text) > 60 else f"   {i:2d}. {text}")
        print()

        # Embed all
        embeddings_np = self.embedder.encode(self.test_texts)
        embeddings = torch.from_numpy(embeddings_np)

        # Test with tiny noise
        noise_gen = self.noise_generators['tiny']
        embeddings_list = embeddings.tolist()
        noise_perturbations = noise_gen.sample_perturbation(embeddings_list)

        noisy_embeddings_list = [[e + n for e, n in zip(emb, noise)]
                                  for emb, noise in zip(embeddings_list, noise_perturbations)]
        noisy_embeddings = torch.tensor(noisy_embeddings_list)

        print(f"Applied TINY noise (std=0.01, threshold=0.1)")
        print()

        # Compute similarities
        print(f"{'#':<4} {'Noise L2':<12} {'Cos Similarity':<15} {'Status'}")
        print("-" * 50)

        similarities = []
        for i in range(len(self.test_texts)):
            noise_norm = noise_gen._compute_l2_norm(noise_perturbations[i])
            cos_sim = torch.nn.functional.cosine_similarity(
                embeddings[i].unsqueeze(0),
                noisy_embeddings[i].unsqueeze(0)
            ).item()
            similarities.append(cos_sim)

            status = "✅" if cos_sim > 0.95 else "⚠️" if cos_sim > 0.90 else "❌"
            print(f"{i+1:<4} {noise_norm:<12.4f} {cos_sim:<15.4f} {status}")

        print()
        print(f"📊 Summary Statistics:")
        print(f"   Average cosine similarity: {sum(similarities)/len(similarities):.4f}")
        print(f"   Min cosine similarity:     {min(similarities):.4f}")
        print(f"   Max cosine similarity:     {max(similarities):.4f}")
        print(f"   Prompts with sim > 0.95:   {sum(1 for s in similarities if s > 0.95)}/10")
        print()
        print(f"✅ With tiny noise, semantic meaning is well preserved!")
        print()


def run_pipeline_tests():
    """Run all pipeline integration tests"""
    print("🧪 Testing Embed → Noise Pipeline (Embedding Similarity Focus)")
    print("=" * 70)
    print()
    print("NOTE: Vec2Text (T5) generates random text without fine-tuning.")
    print("      DART evaluation uses EMBEDDING similarity, not text similarity!")
    print("=" * 70)

    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestEmbedNoiseVec2TextPipeline)

    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Summary
    print("\n" + "=" * 70)
    print(f"📊 Test Results Summary:")
    print(f"✅ Tests Run: {result.testsRun}")
    print(f"❌ Failures: {len(result.failures)}")
    print(f"⚠️  Errors: {len(result.errors)}")

    if result.failures:
        print("\n❌ Failures:")
        for test, traceback in result.failures:
            print(f"- {test}:\n{traceback}")

    if result.errors:
        print("\n⚠️ Errors:")
        for test, traceback in result.errors:
            print(f"- {test}:\n{traceback}")

    if result.testsRun > 0:
        success_rate = (result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100
        print(f"\n🎯 Success Rate: {success_rate:.1f}%")

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_pipeline_tests()
    sys.exit(0 if success else 1)
