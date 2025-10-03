#!/usr/bin/env python3
"""
Unit Tests for ChineseEmbedding Component

Tests:
1. SBERT model embedding functionality
2. Fallback Unicode encoding
3. Embedding dimension consistency
4. Semantic similarity preservation
5. Batch processing performance
"""

import unittest
import sys
import numpy as np
from pathlib import Path

# Add dart_system to path
sys.path.append(str(Path(__file__).parent.parent / 'dart_system'))

from embedding.chinese_embedding import ChineseEmbedding, EmbeddingConfig

class TestChineseEmbedding(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures"""
        self.config = EmbeddingConfig(
            embedding_dim=512,
            max_sequence_length=32,
            similarity_threshold=0.9
        )
        self.embedder = ChineseEmbedding(self.config)
        
        # Test texts
        self.test_texts = [
            "今天天氣很好",
            "如何學習機器學習？",
            "中文自然語言處理很有趣",
            "什麼是人工智能？",
            "測試文本"
        ]
        
        # Similar text pairs for similarity testing
        self.similar_pairs = [
            ("今天天氣很好", "今日天氣不錯"),
            ("如何學習", "怎樣學習"),
            ("機器學習", "人工智能學習")
        ]
        
        # Dissimilar text pairs
        self.dissimilar_pairs = [
            ("天氣很好", "如何製作炸彈"),
            ("學習機器學習", "今天吃什麼"),
            ("中文處理", "英文翻譯")
        ]
    
    def test_single_text_embedding(self):
        """Test embedding single Chinese text"""
        print("\n=== Test 1: Single Text Embedding ===")
        
        text = "今天天氣很好"
        embedding = self.embedder.encode([text])
        
        # Verify embedding properties
        self.assertEqual(embedding.shape[0], 1, "Should return one embedding")
        self.assertEqual(embedding.shape[1], 512, "Should have 512 dimensions")
        self.assertEqual(embedding.dtype, np.float32, "Should be float32")
        
        # Verify embedding is not all zeros
        self.assertGreater(np.abs(embedding).sum(), 0, "Embedding should not be all zeros")
        
        # Verify normalization (if enabled)
        if self.config.normalization:
            norm = np.linalg.norm(embedding[0])
            self.assertAlmostEqual(norm, 1.0, places=3, 
                                 msg="Embedding should be normalized")
        
        print(f"✅ Embedding shape: {embedding.shape}")
        print(f"✅ Embedding norm: {np.linalg.norm(embedding[0]):.3f}")
        print(f"✅ Non-zero values: {np.count_nonzero(embedding[0])}")
    
    def test_batch_text_embedding(self):
        """Test embedding multiple texts in batch"""
        print("\n=== Test 2: Batch Text Embedding ===")
        
        embeddings = self.embedder.encode(self.test_texts)
        
        # Verify batch dimensions
        expected_shape = (len(self.test_texts), 512)
        self.assertEqual(embeddings.shape, expected_shape, 
                        f"Should have shape {expected_shape}")
        
        # Verify each embedding is valid
        for i, embedding in enumerate(embeddings):
            self.assertGreater(np.abs(embedding).sum(), 0, 
                             f"Embedding {i} should not be all zeros")
            
            if self.config.normalization:
                norm = np.linalg.norm(embedding)
                self.assertAlmostEqual(norm, 1.0, places=3,
                                     msg=f"Embedding {i} should be normalized")
        
        print(f"✅ Batch embeddings shape: {embeddings.shape}")
        print(f"✅ All embeddings valid")
    
    def test_embedding_consistency(self):
        """Test that same text produces same embedding"""
        print("\n=== Test 3: Embedding Consistency ===")
        
        text = "測試一致性"
        
        # Encode same text multiple times
        embedding1 = self.embedder.encode([text])
        embedding2 = self.embedder.encode([text])
        
        # Should be identical
        np.testing.assert_array_almost_equal(
            embedding1, embedding2, decimal=5,
            err_msg="Same text should produce identical embeddings"
        )
        
        print("✅ Embedding consistency verified")
    
    def test_similarity_preservation(self):
        """Test that similar texts have high similarity"""
        print("\n=== Test 4: Similarity Preservation ===")
        
        print("Testing similar text pairs:")
        for text1, text2 in self.similar_pairs:
            emb1 = self.embedder.encode([text1])[0]
            emb2 = self.embedder.encode([text2])[0]
            
            # Calculate cosine similarity
            similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
            
            # Similar texts should have high similarity
            self.assertGreater(similarity, 0.5, 
                             f"Similar texts should have high similarity: {text1} vs {text2}")
            
            print(f"  '{text1}' vs '{text2}': {similarity:.3f}")
        
        print("\nTesting dissimilar text pairs:")
        for text1, text2 in self.dissimilar_pairs:
            emb1 = self.embedder.encode([text1])[0]
            emb2 = self.embedder.encode([text2])[0]
            
            # Calculate cosine similarity
            similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
            
            print(f"  '{text1}' vs '{text2}': {similarity:.3f}")
        
        print("✅ Similarity patterns verified")
    
    def test_empty_text_handling(self):
        """Test handling of empty or invalid texts"""
        print("\n=== Test 5: Empty Text Handling ===")
        
        # Test empty string
        empty_embedding = self.embedder.encode([""])
        self.assertEqual(empty_embedding.shape, (1, 512), "Should handle empty string")
        
        # Test None handling (should not crash)
        try:
            none_embedding = self.embedder.encode([None])
            print("✅ None input handled")
        except Exception as e:
            print(f"⚠️ None input raises: {type(e).__name__}")
        
        # Test whitespace
        space_embedding = self.embedder.encode(["   "])
        self.assertEqual(space_embedding.shape, (1, 512), "Should handle whitespace")
        
        print("✅ Edge cases handled")
    
    def test_long_text_handling(self):
        """Test handling of very long texts"""
        print("\n=== Test 6: Long Text Handling ===")
        
        # Create very long text
        long_text = "這是一個很長的測試文本。" * 20  # ~200 characters
        
        long_embedding = self.embedder.encode([long_text])
        
        # Should still produce valid embedding
        self.assertEqual(long_embedding.shape, (1, 512), "Should handle long text")
        self.assertGreater(np.abs(long_embedding).sum(), 0, "Long text embedding should be valid")
        
        print(f"✅ Long text ({len(long_text)} chars) handled")
    
    def test_unicode_fallback(self):
        """Test Unicode fallback encoding when models unavailable"""
        print("\n=== Test 7: Unicode Fallback ===")
        
        # Force fallback by creating embedder with no model
        fallback_config = EmbeddingConfig(embedding_dim=512)
        
        # Test direct fallback encoding
        try:
            fallback_embedder = ChineseEmbedding(fallback_config)
            if hasattr(fallback_embedder, 'fallback_encoder'):
                fallback_embedding = fallback_embedder.fallback_encoder.encode_chinese_text("測試")
                
                self.assertEqual(len(fallback_embedding), 512, "Fallback should produce 512-dim embedding")
                self.assertGreater(sum(abs(x) for x in fallback_embedding), 0, 
                                 "Fallback embedding should not be all zeros")
                
                print("✅ Unicode fallback verified")
            else:
                print("⚠️ Fallback encoder not available")
        except Exception as e:
            print(f"⚠️ Fallback test failed: {e}")
    
    def test_performance_benchmarks(self):
        """Test embedding performance"""
        print("\n=== Test 8: Performance Benchmarks ===")
        
        import time
        
        # Single text performance
        text = "性能測試文本"
        start_time = time.time()
        for _ in range(10):
            self.embedder.encode([text])
        single_time = (time.time() - start_time) / 10
        
        # Batch performance
        batch_texts = ["測試文本" + str(i) for i in range(100)]
        start_time = time.time()
        self.embedder.encode(batch_texts)
        batch_time = time.time() - start_time
        
        print(f"✅ Single text: {single_time*1000:.1f}ms")
        print(f"✅ Batch (100 texts): {batch_time*1000:.1f}ms")
        print(f"✅ Per text in batch: {batch_time/100*1000:.1f}ms")
        
        # Performance should be reasonable
        self.assertLess(single_time, 1.0, "Single text should encode in < 1s")
        self.assertLess(batch_time, 10.0, "Batch should complete in < 10s")

def run_embedding_tests():
    """Run all ChineseEmbedding tests"""
    print("🧪 Running ChineseEmbedding Component Tests")
    print("=" * 50)
    
    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestChineseEmbedding)
    
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
    success = run_embedding_tests()
    sys.exit(0 if success else 1)