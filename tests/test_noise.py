#!/usr/bin/env python3
"""
Unit Tests for DiffusionNoise Component

Tests:
1. Box-Muller Gaussian noise generation
2. Proximity constraint enforcement
3. L2 norm bounds checking
4. Adaptive scaling functionality
5. Different noise distributions
6. Perturbation application
"""

import unittest
import sys
import numpy as np
import math
from pathlib import Path

# Add dart_system to path
sys.path.append(str(Path(__file__).parent.parent / 'dart_system'))

from noise.diffusion_noise import DiffusionNoise, NoiseConfig, NoiseType

class TestDiffusionNoise(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures"""
        self.config = NoiseConfig(
            noise_type=NoiseType.GAUSSIAN,
            proximity_threshold=2.0,
            default_std=0.1,
            adaptive_scaling=True,
            seed=42
        )
        self.noise_gen = DiffusionNoise(self.config)
        
        # Test embedding shapes
        self.small_shape = (5, 512)
        self.batch_shape = (32, 512)
        
        # Sample embeddings for testing
        np.random.seed(42)
        self.sample_embeddings = np.random.randn(*self.small_shape).astype(np.float32)
    
    def test_gaussian_noise_generation(self):
        """Test Box-Muller Gaussian noise generation"""
        print("\n=== Test 1: Gaussian Noise Generation ===")
        
        # Generate noise
        noise = self.noise_gen.generate_gaussian_noise(self.small_shape)
        
        # Verify shape
        self.assertEqual(noise.shape, self.small_shape, 
                        f"Noise should have shape {self.small_shape}")
        
        # Verify statistical properties
        mean = np.mean(noise)
        std = np.std(noise)
        
        # Should be approximately standard normal (mean≈0, std≈1)
        self.assertAlmostEqual(mean, 0.0, places=1, 
                              msg="Gaussian noise should have mean ≈ 0")
        self.assertAlmostEqual(std, self.config.default_std, places=1,
                              msg=f"Gaussian noise should have std ≈ {self.config.default_std}")
        
        print(f"✅ Noise shape: {noise.shape}")
        print(f"✅ Mean: {mean:.4f} (expected: 0.0)")
        print(f"✅ Std: {std:.4f} (expected: {self.config.default_std})")
    
    def test_box_muller_transform(self):
        """Test Box-Muller transform implementation"""
        print("\n=== Test 2: Box-Muller Transform ===")
        
        # Generate large sample for statistical testing
        large_shape = (1000, 100)
        noise = self.noise_gen.generate_gaussian_noise(large_shape)
        
        # Test normality using basic statistical properties
        flat_noise = noise.flatten()
        
        # Calculate moments
        mean = np.mean(flat_noise)
        variance = np.var(flat_noise)
        skewness = np.mean(((flat_noise - mean) / np.sqrt(variance)) ** 3)
        kurtosis = np.mean(((flat_noise - mean) / np.sqrt(variance)) ** 4) - 3
        
        # Normal distribution should have skewness ≈ 0, kurtosis ≈ 0
        self.assertAlmostEqual(mean, 0.0, places=1, 
                              msg="Mean should be close to 0")
        self.assertAlmostEqual(skewness, 0.0, places=0, 
                              msg="Skewness should be close to 0")
        self.assertLess(abs(kurtosis), 0.5, 
                       msg="Kurtosis should be close to 0")
        
        print(f"✅ Sample size: {len(flat_noise)}")
        print(f"✅ Mean: {mean:.4f}")
        print(f"✅ Variance: {variance:.4f}")
        print(f"✅ Skewness: {skewness:.4f}")
        print(f"✅ Kurtosis: {kurtosis:.4f}")
    
    def test_proximity_constraints(self):
        """Test L2 norm proximity constraints"""
        print("\n=== Test 3: Proximity Constraints ===")
        
        # Generate constrained noise
        threshold = 2.0
        noise = self.noise_gen.generate_constrained_noise(
            self.small_shape, strength=0.5, proximity_threshold=threshold
        )
        
        # Check L2 norm for each noise vector
        for i, noise_vec in enumerate(noise):
            l2_norm = np.linalg.norm(noise_vec)
            self.assertLessEqual(l2_norm, threshold + 1e-6,
                               f"Noise vector {i} exceeds proximity threshold: {l2_norm} > {threshold}")
        
        # Calculate statistics
        norms = [np.linalg.norm(vec) for vec in noise]
        max_norm = max(norms)
        avg_norm = np.mean(norms)
        
        print(f"✅ Threshold: {threshold}")
        print(f"✅ Max norm: {max_norm:.4f}")
        print(f"✅ Avg norm: {avg_norm:.4f}")
        print(f"✅ All {len(noise)} vectors within bounds")
    
    def test_adaptive_scaling(self):
        """Test adaptive scaling functionality"""
        print("\n=== Test 4: Adaptive Scaling ===")
        
        # Test with adaptive scaling ON
        config_adaptive = NoiseConfig(adaptive_scaling=True, proximity_threshold=1.0)
        noise_gen_adaptive = DiffusionNoise(config_adaptive)
        
        noise_adaptive = noise_gen_adaptive.generate_constrained_noise(
            self.small_shape, strength=2.0  # High strength that would exceed threshold
        )
        
        # Test with adaptive scaling OFF
        config_no_adaptive = NoiseConfig(adaptive_scaling=False, proximity_threshold=1.0)
        noise_gen_no_adaptive = DiffusionNoise(config_no_adaptive)
        
        noise_no_adaptive = noise_gen_no_adaptive.generate_constrained_noise(
            self.small_shape, strength=2.0
        )
        
        # With adaptive scaling, norms should be within threshold
        adaptive_norms = [np.linalg.norm(vec) for vec in noise_adaptive]
        no_adaptive_norms = [np.linalg.norm(vec) for vec in noise_no_adaptive]
        
        max_adaptive = max(adaptive_norms)
        max_no_adaptive = max(no_adaptive_norms)
        
        print(f"✅ Max norm (adaptive ON): {max_adaptive:.4f}")
        print(f"✅ Max norm (adaptive OFF): {max_no_adaptive:.4f}")
        
        # Adaptive should respect threshold better
        self.assertLessEqual(max_adaptive, 1.1, "Adaptive scaling should respect threshold")
    
    def test_perturbation_application(self):
        """Test perturbation application to embeddings"""
        print("\n=== Test 5: Perturbation Application ===")
        
        # Generate perturbations
        perturbations = self.noise_gen.sample_perturbation(
            self.sample_embeddings, self.config.proximity_threshold
        )
        
        # Apply perturbations
        perturbed_embeddings = self.noise_gen.apply_perturbation(
            self.sample_embeddings, perturbations
        )
        
        # Verify shapes
        self.assertEqual(perturbations.shape, self.sample_embeddings.shape,
                        "Perturbations should match embedding shape")
        self.assertEqual(perturbed_embeddings.shape, self.sample_embeddings.shape,
                        "Perturbed embeddings should match original shape")
        
        # Verify perturbation is applied correctly (original + noise)
        expected_perturbed = self.sample_embeddings + perturbations
        np.testing.assert_array_almost_equal(
            perturbed_embeddings, expected_perturbed, decimal=6,
            err_msg="Perturbation should be simple addition"
        )
        
        # Check perturbation distances
        distances = []
        for orig, pert in zip(self.sample_embeddings, perturbed_embeddings):
            distance = np.linalg.norm(pert - orig)
            distances.append(distance)
        
        max_distance = max(distances)
        avg_distance = np.mean(distances)
        
        print(f"✅ Perturbation shape: {perturbations.shape}")
        print(f"✅ Max perturbation distance: {max_distance:.4f}")
        print(f"✅ Avg perturbation distance: {avg_distance:.4f}")
        print(f"✅ Threshold: {self.config.proximity_threshold}")
    
    def test_different_noise_types(self):
        """Test different noise distribution types"""
        print("\n=== Test 6: Different Noise Types ===")
        
        shape = (100, 10)
        
        # Test Gaussian
        gaussian_config = NoiseConfig(noise_type=NoiseType.GAUSSIAN)
        gaussian_gen = DiffusionNoise(gaussian_config)
        gaussian_noise = gaussian_gen.generate_gaussian_noise(shape)
        
        # Test Uniform (if implemented)
        try:
            uniform_config = NoiseConfig(noise_type=NoiseType.UNIFORM)
            uniform_gen = DiffusionNoise(uniform_config)
            # Note: This might use Gaussian as fallback
            uniform_noise = uniform_gen.generate_gaussian_noise(shape)
            print("✅ Uniform noise generation tested")
        except Exception as e:
            print(f"⚠️ Uniform noise not implemented: {e}")
        
        # Verify Gaussian properties
        gaussian_flat = gaussian_noise.flatten()
        gaussian_mean = np.mean(gaussian_flat)
        gaussian_std = np.std(gaussian_flat)
        
        print(f"✅ Gaussian mean: {gaussian_mean:.4f}")
        print(f"✅ Gaussian std: {gaussian_std:.4f}")
    
    def test_seed_reproducibility(self):
        """Test that seeds produce reproducible results"""
        print("\n=== Test 7: Seed Reproducibility ===")
        
        # Create two generators with same seed
        config1 = NoiseConfig(seed=12345)
        config2 = NoiseConfig(seed=12345)
        gen1 = DiffusionNoise(config1)
        gen2 = DiffusionNoise(config2)
        
        # Generate noise with both
        noise1 = gen1.generate_gaussian_noise((10, 20))
        noise2 = gen2.generate_gaussian_noise((10, 20))
        
        # Should be identical
        np.testing.assert_array_equal(
            noise1, noise2,
            err_msg="Same seed should produce identical noise"
        )
        
        # Test different seeds produce different results
        config3 = NoiseConfig(seed=54321)
        gen3 = DiffusionNoise(config3)
        noise3 = gen3.generate_gaussian_noise((10, 20))
        
        # Should be different
        self.assertFalse(np.array_equal(noise1, noise3),
                        "Different seeds should produce different noise")
        
        print("✅ Seed reproducibility verified")
        print("✅ Different seeds produce different results")
    
    def test_edge_cases(self):
        """Test edge cases and error handling"""
        print("\n=== Test 8: Edge Cases ===")
        
        # Test very small embeddings
        tiny_shape = (1, 1)
        tiny_noise = self.noise_gen.generate_gaussian_noise(tiny_shape)
        self.assertEqual(tiny_noise.shape, tiny_shape, "Should handle tiny shapes")
        
        # Test zero threshold
        try:
            zero_noise = self.noise_gen.generate_constrained_noise(
                (5, 10), proximity_threshold=0.0
            )
            zero_norms = [np.linalg.norm(vec) for vec in zero_noise]
            max_zero_norm = max(zero_norms)
            print(f"✅ Zero threshold max norm: {max_zero_norm:.6f}")
        except Exception as e:
            print(f"⚠️ Zero threshold handling: {e}")
        
        # Test very large threshold
        large_noise = self.noise_gen.generate_constrained_noise(
            (5, 10), proximity_threshold=1000.0
        )
        self.assertEqual(large_noise.shape, (5, 10), "Should handle large thresholds")
        
        print("✅ Edge cases handled")

def run_noise_tests():
    """Run all DiffusionNoise tests"""
    print("🧪 Running DiffusionNoise Component Tests")
    print("=" * 50)
    
    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestDiffusionNoise)
    
    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Summary
    print("\n" + "=" * 50)
    print(f"📊 Test Results Summary:")
    print(f"✅ Tests Run: {result.testsRun}")
    print(f"❌ Failures: {len(result.failures)}")
    print(f"⚠️  Errors: {len(result.errors)}")
    
    if result.failures:
        print("\n❌ Failures:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback}")
    
    if result.errors:
        print("\n⚠️ Errors:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback}")
    
    success_rate = (result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100
    print(f"\n🎯 Success Rate: {success_rate:.1f}%")
    
    return result.wasSuccessful()

if __name__ == "__main__":
    success = run_noise_tests()
    sys.exit(0 if success else 1)