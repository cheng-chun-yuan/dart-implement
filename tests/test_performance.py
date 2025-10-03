#!/usr/bin/env python3
"""
Performance Benchmarking Tests for DART System

Tests:
1. Individual component performance
2. End-to-end throughput testing
3. Memory usage monitoring
4. Scalability testing
5. Latency measurements
6. Resource utilization profiling
"""

import unittest
import sys
import time
import psutil
import os
import tempfile
import csv
from pathlib import Path
import gc

# Add dart_system to path
sys.path.append(str(Path(__file__).parent.parent / 'dart_system'))

from core.dart_controller import DARTController, DARTConfig
from data.data_loader import ChineseDataLoader
from embedding.chinese_embedding import ChineseEmbedding, EmbeddingConfig
from noise.diffusion_noise import DiffusionNoise, NoiseConfig
from reconstruction.vec2text import ChineseVec2Text, ReconstructionConfig

class TestDARTPerformance(unittest.TestCase):
    
    def setUp(self):
        """Set up performance test fixtures"""
        # Create temporary test data
        self.temp_dir = tempfile.mkdtemp()
        self.test_csv = os.path.join(self.temp_dir, 'perf_test.csv')
        
        # Create larger test dataset
        with open(self.test_csv, 'w', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['有害問題'])
            for i in range(100):
                writer.writerow([f'測試有害內容 {i}: 如何進行不當行為？這是一個較長的測試文本來評估性能。'])
        
        # Standard configuration
        self.config = DARTConfig(
            csv_path=self.test_csv,
            embedding_dim=512,
            max_texts_per_batch=8,
            proximity_threshold=2.0,
            noise_std=0.1,
            verbose=False,
            random_seed=42
        )
        
        # Performance test texts of varying lengths
        self.short_texts = [f"短文本{i}" for i in range(10)]
        self.medium_texts = [f"這是一個中等長度的測試文本 {i}，用於評估系統性能" for i in range(10)]
        self.long_texts = [f"這是一個很長的測試文本 {i}。" * 10 for i in range(10)]
    
    def tearDown(self):
        """Clean up performance test fixtures"""
        import shutil
        shutil.rmtree(self.temp_dir)
        gc.collect()  # Force garbage collection
    
    def measure_memory_usage(self, func, *args, **kwargs):
        """Measure memory usage of a function"""
        process = psutil.Process(os.getpid())
        
        # Get baseline memory
        baseline_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Execute function
        start_time = time.time()
        result = func(*args, **kwargs)
        execution_time = time.time() - start_time
        
        # Get peak memory
        peak_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_delta = peak_memory - baseline_memory
        
        return result, execution_time, memory_delta, peak_memory
    
    def test_data_loader_performance(self):
        """Test data loading performance"""
        print("\n=== Performance Test 1: Data Loading ===")
        
        loader = ChineseDataLoader(self.test_csv)
        
        # Measure loading performance
        def load_data():
            return loader.load_csv_dataset()
        
        result, exec_time, memory_delta, peak_memory = self.measure_memory_usage(load_data)
        
        # Performance metrics
        texts_per_second = len(result) / exec_time if exec_time > 0 else 0
        memory_per_text = memory_delta / len(result) if len(result) > 0 else 0
        
        print(f"✅ Texts loaded: {len(result)}")
        print(f"✅ Loading time: {exec_time:.3f}s")
        print(f"✅ Texts/second: {texts_per_second:.1f}")
        print(f"✅ Memory delta: {memory_delta:.1f}MB")
        print(f"✅ Memory/text: {memory_per_text:.3f}MB")
        
        # Performance assertions
        self.assertLess(exec_time, 2.0, "Data loading should complete in < 2s")
        self.assertGreater(texts_per_second, 50, "Should load > 50 texts/second")
        self.assertLess(memory_delta, 50, "Memory delta should be < 50MB")
    
    def test_embedding_performance(self):
        """Test embedding performance with different text lengths"""
        print("\n=== Performance Test 2: Embedding ===")
        
        config = EmbeddingConfig(embedding_dim=512)
        embedder = ChineseEmbedding(config)
        
        # Test different text lengths
        test_cases = [
            ("Short texts", self.short_texts),
            ("Medium texts", self.medium_texts),
            ("Long texts", self.long_texts[:5])  # Fewer long texts to save time
        ]
        
        for case_name, texts in test_cases:
            print(f"\n--- {case_name} ({len(texts)} texts) ---")
            
            def embed_texts():
                return embedder.encode(texts)
            
            embeddings, exec_time, memory_delta, peak_memory = self.measure_memory_usage(embed_texts)
            
            # Performance metrics
            texts_per_second = len(texts) / exec_time if exec_time > 0 else 0
            time_per_text = exec_time / len(texts) if len(texts) > 0 else 0
            
            print(f"  Embedding shape: {embeddings.shape}")
            print(f"  Total time: {exec_time:.3f}s")
            print(f"  Time per text: {time_per_text*1000:.1f}ms")
            print(f"  Texts/second: {texts_per_second:.1f}")
            print(f"  Memory delta: {memory_delta:.1f}MB")
            
            # Verify output
            self.assertEqual(embeddings.shape[0], len(texts), "Should embed all texts")
            self.assertEqual(embeddings.shape[1], 512, "Should have correct dimensions")
            
            # Performance expectations
            self.assertLess(time_per_text, 1.0, f"Each text should embed in < 1s: {case_name}")
    
    def test_noise_generation_performance(self):
        """Test noise generation performance"""
        print("\n=== Performance Test 3: Noise Generation ===")
        
        config = NoiseConfig(proximity_threshold=2.0, default_std=0.1)
        noise_gen = DiffusionNoise(config)
        
        # Test different batch sizes
        batch_sizes = [1, 10, 50, 100]
        
        for batch_size in batch_sizes:
            shape = (batch_size, 512)
            print(f"\n--- Batch size: {batch_size} ---")
            
            def generate_noise():
                return noise_gen.generate_constrained_noise(shape, strength=0.1)
            
            noise, exec_time, memory_delta, peak_memory = self.measure_memory_usage(generate_noise)
            
            # Performance metrics
            vectors_per_second = batch_size / exec_time if exec_time > 0 else 0
            time_per_vector = exec_time / batch_size if batch_size > 0 else 0
            
            print(f"  Noise shape: {noise.shape}")
            print(f"  Total time: {exec_time:.3f}s")
            print(f"  Time per vector: {time_per_vector*1000:.1f}ms")
            print(f"  Vectors/second: {vectors_per_second:.1f}")
            print(f"  Memory delta: {memory_delta:.1f}MB")
            
            # Verify output
            self.assertEqual(noise.shape, shape, "Should generate correct shape")
            
            # Performance expectations
            self.assertLess(time_per_vector, 0.1, "Each vector should generate in < 100ms")
    
    def test_reconstruction_performance(self):
        """Test text reconstruction performance"""
        print("\n=== Performance Test 4: Reconstruction ===")
        
        config = ReconstructionConfig(max_length=64, max_iters=3)
        reconstructor = ChineseVec2Text(config)
        
        # Create test embeddings and texts
        import numpy as np
        np.random.seed(42)
        
        test_cases = [
            ("Small batch", self.short_texts[:5], np.random.randn(5, 512).astype(np.float32)),
            ("Medium batch", self.medium_texts[:10], np.random.randn(10, 512).astype(np.float32)),
            ("Large batch", self.short_texts * 5, np.random.randn(50, 512).astype(np.float32))
        ]
        
        for case_name, texts, embeddings in test_cases:
            print(f"\n--- {case_name} ({len(texts)} texts) ---")
            
            def reconstruct_texts():
                return reconstructor.decode(embeddings, texts)
            
            reconstructed, exec_time, memory_delta, peak_memory = self.measure_memory_usage(reconstruct_texts)
            
            # Performance metrics
            texts_per_second = len(texts) / exec_time if exec_time > 0 else 0
            time_per_text = exec_time / len(texts) if len(texts) > 0 else 0
            
            print(f"  Texts reconstructed: {len(reconstructed)}")
            print(f"  Total time: {exec_time:.3f}s")
            print(f"  Time per text: {time_per_text*1000:.1f}ms")
            print(f"  Texts/second: {texts_per_second:.1f}")
            print(f"  Memory delta: {memory_delta:.1f}MB")
            
            # Verify output
            self.assertEqual(len(reconstructed), len(texts), "Should reconstruct all texts")
            
            # Performance expectations
            self.assertLess(time_per_text, 2.0, f"Each reconstruction should take < 2s: {case_name}")
    
    def test_end_to_end_performance(self):
        """Test complete end-to-end performance"""
        print("\n=== Performance Test 5: End-to-End Performance ===")
        
        controller = DARTController(self.config)
        
        # Test different text batch sizes
        batch_sizes = [1, 5, 10, 20]
        
        for batch_size in batch_sizes:
            test_texts = [f"端到端測試文本 {i}" for i in range(batch_size)]
            print(f"\n--- Batch size: {batch_size} ---")
            
            def end_to_end_attack():
                return controller.run_attack(test_texts)
            
            result, exec_time, memory_delta, peak_memory = self.measure_memory_usage(end_to_end_attack)
            
            # Performance metrics
            if result["success"]:
                texts_per_second = batch_size / exec_time if exec_time > 0 else 0
                time_per_text = exec_time / batch_size if batch_size > 0 else 0
                avg_similarity = result.get("avg_similarity", 0)
                
                print(f"  Success: True")
                print(f"  Total time: {exec_time:.3f}s")
                print(f"  Time per text: {time_per_text*1000:.1f}ms")
                print(f"  Texts/second: {texts_per_second:.1f}")
                print(f"  Avg similarity: {avg_similarity:.3f}")
                print(f"  Memory delta: {memory_delta:.1f}MB")
                print(f"  Peak memory: {peak_memory:.1f}MB")
                
                # Performance expectations
                self.assertLess(time_per_text, 5.0, f"Each E2E attack should take < 5s: batch {batch_size}")
                self.assertGreater(avg_similarity, 0.5, "Should maintain reasonable similarity")
            else:
                print(f"  Success: False")
                print(f"  Error: {result.get('error', 'Unknown')}")
    
    def test_batch_processing_scalability(self):
        """Test batch processing scalability"""
        print("\n=== Performance Test 6: Batch Processing Scalability ===")
        
        controller = DARTController(self.config)
        
        # Test increasing batch sizes
        batch_sizes = [10, 25, 50]
        
        performance_data = []
        
        for batch_size in batch_sizes:
            print(f"\n--- Testing batch size: {batch_size} ---")
            
            def batch_attack():
                return controller.run_batch_attack(batch_size=8)  # Use fixed internal batch size
            
            batch_results, exec_time, memory_delta, peak_memory = self.measure_memory_usage(batch_attack)
            
            # Calculate total texts processed
            total_texts = sum(
                len(batch.get("original_texts", [])) 
                for batch in batch_results 
                if batch.get("success", False)
            )
            
            # Performance metrics
            if total_texts > 0:
                texts_per_second = total_texts / exec_time if exec_time > 0 else 0
                time_per_text = exec_time / total_texts
                memory_per_text = memory_delta / total_texts
                
                performance_data.append({
                    "batch_size": batch_size,
                    "total_texts": total_texts,
                    "exec_time": exec_time,
                    "texts_per_second": texts_per_second,
                    "time_per_text": time_per_text,
                    "memory_delta": memory_delta,
                    "memory_per_text": memory_per_text
                })
                
                print(f"  Total texts: {total_texts}")
                print(f"  Total time: {exec_time:.3f}s")
                print(f"  Time per text: {time_per_text*1000:.1f}ms")
                print(f"  Texts/second: {texts_per_second:.1f}")
                print(f"  Memory delta: {memory_delta:.1f}MB")
                print(f"  Memory per text: {memory_per_text:.3f}MB")
            else:
                print(f"  No texts processed successfully")
        
        # Analyze scalability
        if len(performance_data) >= 2:
            print(f"\n--- Scalability Analysis ---")
            for i in range(1, len(performance_data)):
                prev = performance_data[i-1]
                curr = performance_data[i]
                
                throughput_ratio = curr["texts_per_second"] / prev["texts_per_second"]
                memory_ratio = curr["memory_per_text"] / prev["memory_per_text"]
                
                print(f"  {prev['batch_size']} → {curr['batch_size']} texts:")
                print(f"    Throughput ratio: {throughput_ratio:.2f}x")
                print(f"    Memory ratio: {memory_ratio:.2f}x")
    
    def test_memory_stress(self):
        """Test memory usage under stress"""
        print("\n=== Performance Test 7: Memory Stress Test ===")
        
        controller = DARTController(self.config)
        process = psutil.Process(os.getpid())
        
        # Record initial memory
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        print(f"Initial memory: {initial_memory:.1f}MB")
        
        # Perform multiple attacks to stress memory
        num_iterations = 10
        test_texts = ["記憶體壓力測試文本"] * 5
        
        max_memory = initial_memory
        
        for i in range(num_iterations):
            result = controller.run_attack(test_texts)
            
            current_memory = process.memory_info().rss / 1024 / 1024
            max_memory = max(max_memory, current_memory)
            
            if i % 3 == 0:  # Report every 3rd iteration
                print(f"  Iteration {i+1}: {current_memory:.1f}MB")
            
            # Force garbage collection occasionally
            if i % 5 == 0:
                gc.collect()
        
        final_memory = process.memory_info().rss / 1024 / 1024
        memory_growth = final_memory - initial_memory
        peak_memory_growth = max_memory - initial_memory
        
        print(f"Final memory: {final_memory:.1f}MB")
        print(f"Memory growth: {memory_growth:.1f}MB")
        print(f"Peak memory growth: {peak_memory_growth:.1f}MB")
        
        # Memory should not grow excessively
        self.assertLess(memory_growth, 100, "Memory growth should be < 100MB")
        self.assertLess(peak_memory_growth, 200, "Peak memory growth should be < 200MB")

def run_performance_tests():
    """Run all performance tests"""
    print("🚀 Running DART Performance Benchmarks")
    print("=" * 50)
    print("⚠️  Note: Performance tests may take several minutes to complete")
    print()
    
    # System information
    print("System Information:")
    print(f"  CPU cores: {psutil.cpu_count()}")
    print(f"  RAM: {psutil.virtual_memory().total / 1024 / 1024 / 1024:.1f}GB")
    print(f"  Python: {sys.version.split()[0]}")
    print()
    
    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestDARTPerformance)
    
    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Summary
    print("\n" + "=" * 50)
    print(f"📊 Performance Test Results:")
    print(f"✅ Tests Run: {result.testsRun}")
    print(f"❌ Failures: {len(result.failures)}")
    print(f"⚠️  Errors: {len(result.errors)}")
    
    if result.failures:
        print("\n❌ Performance Issues:")
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
    success = run_performance_tests()
    sys.exit(0 if success else 1)