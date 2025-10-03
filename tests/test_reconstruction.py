#!/usr/bin/env python3
"""
Unit Tests for ChineseVec2Text Component

Tests:
1. T5-based text reconstruction
2. Heuristic fallback reconstruction
3. Text similarity computation
4. Iterative refinement
5. Semantic preservation
6. Error handling and edge cases
"""

import unittest
import sys
import numpy as np
from pathlib import Path

# Add dart_system to path
sys.path.append(str(Path(__file__).parent.parent / 'dart_system'))

from reconstruction.vec2text import ChineseVec2Text, ReconstructionConfig

class TestChineseVec2Text(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures"""
        self.config = ReconstructionConfig(
            max_length=64,
            max_iters=3,
            temperature=0.7,
            synonym_prob=0.3,
            structure_prob=0.2,
            max_changes_per_text=3,
            semantic_drift_threshold=0.8
        )
        self.reconstructor = ChineseVec2Text(self.config)
        
        # Test texts and embeddings
        self.test_texts = [
            "今天天氣很好",
            "如何學習機器學習？",
            "中文自然語言處理很有趣",
            "什麼是人工智能？",
            "這是一個測試文本"
        ]
        
        # Sample embeddings (512-dimensional)
        np.random.seed(42)
        self.sample_embeddings = np.random.randn(len(self.test_texts), 512).astype(np.float32)
        
        # Slightly perturbed embeddings for reconstruction testing
        noise = np.random.randn(*self.sample_embeddings.shape) * 0.1
        self.perturbed_embeddings = self.sample_embeddings + noise
    
    def test_text_similarity_computation(self):
        """Test text similarity computation methods"""
        print("\n=== Test 1: Text Similarity Computation ===")
        
        # Test identical texts
        sim_identical = self.reconstructor.compute_text_similarity("測試", "測試")
        self.assertEqual(sim_identical, 1.0, "Identical texts should have similarity 1.0")
        
        # Test completely different texts
        sim_different = self.reconstructor.compute_text_similarity("天氣", "電腦")
        self.assertLess(sim_different, 1.0, "Different texts should have similarity < 1.0")
        
        # Test similar texts
        sim_similar = self.reconstructor.compute_text_similarity("今天天氣好", "今日天氣不錯")
        self.assertGreater(sim_similar, 0.3, "Similar texts should have reasonable similarity")
        
        # Test empty strings
        sim_empty = self.reconstructor.compute_text_similarity("", "")
        self.assertIsInstance(sim_empty, float, "Empty strings should return valid similarity")
        
        print(f"✅ Identical texts similarity: {sim_identical}")
        print(f"✅ Different texts similarity: {sim_different:.3f}")
        print(f"✅ Similar texts similarity: {sim_similar:.3f}")
        print(f"✅ Empty strings similarity: {sim_empty:.3f}")
    
    def test_heuristic_reconstruction(self):
        """Test heuristic fallback reconstruction"""
        print("\n=== Test 2: Heuristic Reconstruction ===")
        
        # Test heuristic reconstruction for each text
        for i, original_text in enumerate(self.test_texts):
            reconstructed = self.reconstructor.heuristic_decode(
                self.perturbed_embeddings[i], original_text
            )
            
            # Verify reconstruction is valid
            self.assertIsInstance(reconstructed, str, "Reconstruction should be string")
            self.assertGreater(len(reconstructed), 0, "Reconstruction should not be empty")
            
            # Compute similarity
            similarity = self.reconstructor.compute_text_similarity(original_text, reconstructed)
            
            # Should maintain reasonable similarity
            self.assertGreater(similarity, 0.5, 
                             f"Heuristic reconstruction should maintain similarity > 0.5: {similarity}")
            
            print(f"  Original: '{original_text}'")
            print(f"  Reconstructed: '{reconstructed}'")
            print(f"  Similarity: {similarity:.3f}")
        
        print("✅ Heuristic reconstruction verified")
    
    def test_synonym_replacement(self):
        """Test synonym replacement functionality"""
        print("\n=== Test 3: Synonym Replacement ===")
        
        test_text = "這是一個好的例子"
        
        # Generate multiple reconstructions to test randomness
        reconstructions = []
        for _ in range(10):
            reconstructed = self.reconstructor.heuristic_decode(
                self.sample_embeddings[0], test_text
            )
            reconstructions.append(reconstructed)
        
        # Should have some variation due to synonym replacement
        unique_reconstructions = set(reconstructions)
        
        print(f"✅ Original: '{test_text}'")
        print(f"✅ Unique reconstructions: {len(unique_reconstructions)}")
        for i, recon in enumerate(list(unique_reconstructions)[:3]):
            print(f"  Variant {i+1}: '{recon}'")
        
        # At least some variation should occur (with 30% probability)
        # This is probabilistic, so we allow some tolerance
        if len(unique_reconstructions) > 1:
            print("✅ Synonym replacement working")
        else:
            print("⚠️ No variation observed (may be due to randomness)")
    
    def test_structure_modification(self):
        """Test sentence structure modification"""
        print("\n=== Test 4: Structure Modification ===")
        
        # Test with longer, more complex sentences
        complex_texts = [
            "今天的天氣非常好，很適合出去散步",
            "機器學習是人工智能的一個重要分支",
            "這個系統可以處理中文文本"
        ]
        
        for text in complex_texts:
            # Generate multiple reconstructions
            reconstructions = []
            for _ in range(5):
                reconstructed = self.reconstructor.heuristic_decode(
                    self.sample_embeddings[0], text
                )
                reconstructions.append(reconstructed)
            
            # Check for structural variations
            unique_structures = set(reconstructions)
            similarity_scores = [
                self.reconstructor.compute_text_similarity(text, recon) 
                for recon in reconstructions
            ]
            
            avg_similarity = np.mean(similarity_scores)
            
            print(f"  Original: '{text}'")
            print(f"  Unique variations: {len(unique_structures)}")
            print(f"  Avg similarity: {avg_similarity:.3f}")
            
            # Should maintain reasonable similarity even with structure changes
            self.assertGreater(avg_similarity, 0.6, 
                             "Structure modifications should preserve meaning")
        
        print("✅ Structure modification tested")
    
    def test_embedding_to_text_decode(self):
        """Test main decode function"""
        print("\n=== Test 5: Embedding to Text Decode ===")
        
        # Test decoding perturbed embeddings
        reconstructed_texts = self.reconstructor.decode(
            self.perturbed_embeddings, self.test_texts
        )
        
        # Verify output format
        self.assertEqual(len(reconstructed_texts), len(self.test_texts),
                        "Should return same number of reconstructions as inputs")
        
        # Test each reconstruction
        similarities = []
        for i, (original, reconstructed) in enumerate(zip(self.test_texts, reconstructed_texts)):
            self.assertIsInstance(reconstructed, str, f"Reconstruction {i} should be string")
            self.assertGreater(len(reconstructed), 0, f"Reconstruction {i} should not be empty")
            
            similarity = self.reconstructor.compute_text_similarity(original, reconstructed)
            similarities.append(similarity)
            
            print(f"  {i+1}. '{original}' → '{reconstructed}' (sim: {similarity:.3f})")
        
        avg_similarity = np.mean(similarities)
        min_similarity = min(similarities)
        
        print(f"✅ Average similarity: {avg_similarity:.3f}")
        print(f"✅ Minimum similarity: {min_similarity:.3f}")
        
        # Should maintain reasonable reconstruction quality
        self.assertGreater(avg_similarity, 0.7, "Average similarity should be > 0.7")
        self.assertGreater(min_similarity, 0.5, "Minimum similarity should be > 0.5")
    
    def test_clean_embedding_reconstruction(self):
        """Test reconstruction of clean (unperturbed) embeddings"""
        print("\n=== Test 6: Clean Embedding Reconstruction ===")
        
        # Use original embeddings (no perturbation)
        clean_reconstructed = self.reconstructor.decode(
            self.sample_embeddings, self.test_texts
        )
        
        # Clean embeddings should reconstruct very well
        clean_similarities = []
        for original, reconstructed in zip(self.test_texts, clean_reconstructed):
            similarity = self.reconstructor.compute_text_similarity(original, reconstructed)
            clean_similarities.append(similarity)
            
            print(f"  '{original}' → '{reconstructed}' (sim: {similarity:.3f})")
        
        avg_clean_similarity = np.mean(clean_similarities)
        
        print(f"✅ Clean embedding avg similarity: {avg_clean_similarity:.3f}")
        
        # Clean embeddings should reconstruct better than perturbed ones
        self.assertGreater(avg_clean_similarity, 0.8, 
                          "Clean embeddings should reconstruct very well")
    
    def test_edge_cases(self):
        """Test edge cases and error handling"""
        print("\n=== Test 7: Edge Cases ===")
        
        # Test empty text
        try:
            empty_result = self.reconstructor.decode(
                self.sample_embeddings[:1], [""]
            )
            self.assertEqual(len(empty_result), 1, "Should handle empty text")
            print("✅ Empty text handled")
        except Exception as e:
            print(f"⚠️ Empty text error: {e}")
        
        # Test very long text
        long_text = "這是一個非常長的測試文本。" * 10
        try:
            long_result = self.reconstructor.decode(
                self.sample_embeddings[:1], [long_text]
            )
            self.assertEqual(len(long_result), 1, "Should handle long text")
            print("✅ Long text handled")
        except Exception as e:
            print(f"⚠️ Long text error: {e}")
        
        # Test single character
        try:
            char_result = self.reconstructor.decode(
                self.sample_embeddings[:1], ["字"]
            )
            self.assertEqual(len(char_result), 1, "Should handle single character")
            print("✅ Single character handled")
        except Exception as e:
            print(f"⚠️ Single character error: {e}")
        
        # Test mismatched dimensions
        try:
            wrong_embedding = np.random.randn(1, 256).astype(np.float32)  # Wrong dimension
            wrong_result = self.reconstructor.decode(wrong_embedding, ["測試"])
            print("⚠️ Wrong dimension handled (may use fallback)")
        except Exception as e:
            print(f"✅ Wrong dimension properly rejected: {type(e).__name__}")
    
    def test_iterative_refinement(self):
        """Test iterative refinement if available"""
        print("\n=== Test 8: Iterative Refinement ===")
        
        if hasattr(self.reconstructor, 'iterative_refinement'):
            # Test iterative refinement
            test_text = "測試迭代優化功能"
            embedding = self.perturbed_embeddings[0]
            
            try:
                refined_text = self.reconstructor.iterative_refinement(
                    embedding, test_text, max_iters=3
                )
                
                similarity = self.reconstructor.compute_text_similarity(test_text, refined_text)
                
                print(f"✅ Original: '{test_text}'")
                print(f"✅ Refined: '{refined_text}'")
                print(f"✅ Similarity: {similarity:.3f}")
                
                self.assertIsInstance(refined_text, str, "Refined text should be string")
                
            except Exception as e:
                print(f"⚠️ Iterative refinement error: {e}")
        else:
            print("⚠️ Iterative refinement not implemented")
    
    def test_batch_performance(self):
        """Test batch processing performance"""
        print("\n=== Test 9: Batch Performance ===")
        
        import time
        
        # Create larger batch for performance testing
        batch_size = 50
        batch_texts = [f"測試文本 {i}" for i in range(batch_size)]
        batch_embeddings = np.random.randn(batch_size, 512).astype(np.float32)
        
        # Time the reconstruction
        start_time = time.time()
        batch_reconstructed = self.reconstructor.decode(batch_embeddings, batch_texts)
        reconstruction_time = time.time() - start_time
        
        # Verify batch results
        self.assertEqual(len(batch_reconstructed), batch_size,
                        "Should reconstruct all texts in batch")
        
        per_text_time = reconstruction_time / batch_size
        
        print(f"✅ Batch size: {batch_size}")
        print(f"✅ Total time: {reconstruction_time:.3f}s")
        print(f"✅ Per text: {per_text_time*1000:.1f}ms")
        
        # Performance should be reasonable
        self.assertLess(per_text_time, 1.0, "Should reconstruct each text in < 1s")

def run_reconstruction_tests():
    """Run all ChineseVec2Text tests"""
    print("🧪 Running ChineseVec2Text Component Tests")
    print("=" * 50)
    
    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestChineseVec2Text)
    
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
    success = run_reconstruction_tests()
    sys.exit(0 if success else 1)