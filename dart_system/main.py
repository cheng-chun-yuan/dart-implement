"""
DART系統主入口文件
統一的命令行接口，整合所有模組功能

使用方式:
    python main.py --mode single --texts "測試文本1" "測試文本2"
    python main.py --mode batch --csv-path problem.csv --sample-size 100
    python main.py --mode test --comprehensive
"""

import argparse
import sys
import logging
import json
from pathlib import Path
from typing import List, Optional

# 添加模組路徑
sys.path.append(str(Path(__file__).parent))

from core.dart_controller import DARTController, DARTConfig

def setup_logging(level: str = "INFO"):
    """設置日誌配置"""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('dart_system.log', encoding='utf-8')
        ]
    )

def run_single_attack(args) -> None:
    """執行單次攻擊"""
    print("=== DART單次攻擊模式 ===")
    
    # 創建配置
    config = DARTConfig(
        csv_path=args.csv_path,
        embedding_dim=args.embedding_dim,
        proximity_threshold=args.proximity_threshold,
        noise_std=args.noise_std,
        verbose=args.verbose
    )
    
    # 創建控制器
    controller = DARTController(config)
    
    # 執行攻擊
    results = controller.run_attack(args.texts)
    
    # 顯示結果
    if results["success"]:
        print(f"\n處理了 {len(args.texts)} 個文本，耗時 {results['processing_time']:.2f}秒")
        print(f"平均相似度: {results['avg_similarity']:.3f}")
        
        for i, (orig, recon, sim) in enumerate(zip(
            results["original_texts"],
            results["reconstructed_texts"],
            results["similarities"]
        )):
            print(f"\n--- 文本 {i+1} ---")
            print(f"原始: {orig}")
            print(f"重建: {recon}")
            print(f"相似度: {sim:.3f}")
        
        # 評估效果
        evaluation = controller.evaluate_attack_effectiveness(results)
        print(f"\n=== 攻擊效果評估 ===")
        for key, value in evaluation.items():
            if isinstance(value, float):
                print(f"{key}: {value:.3f}")
    
    else:
        print(f"攻擊失敗: {results.get('error', 'Unknown error')}")

def run_batch_attack(args) -> None:
    """執行批次攻擊"""
    print("=== DART批次攻擊模式 ===")
    
    config = DARTConfig(
        csv_path=args.csv_path,
        embedding_dim=args.embedding_dim,
        proximity_threshold=args.proximity_threshold,
        noise_std=args.noise_std,
        max_texts_per_batch=args.batch_size,
        verbose=args.verbose
    )
    
    controller = DARTController(config)
    
    # 載入數據集
    dataset = controller.load_dataset(args.sample_size)
    if dataset["total"] == 0:
        print("錯誤: 無法載入數據集")
        return
    
    print(f"載入數據集: {dataset['total']} 個文本")
    print(f"有害文本: {len(dataset['harmful'])}")
    print(f"良性文本: {len(dataset['benign'])}")
    
    # 執行批次攻擊
    batch_results = controller.run_batch_attack(args.batch_size)
    
    # 統計結果
    total_processed = 0
    total_successful = 0
    avg_similarities = []
    
    for batch_result in batch_results:
        if batch_result["success"]:
            total_processed += len(batch_result["original_texts"])
            total_successful += len(batch_result["original_texts"])
            avg_similarities.append(batch_result["avg_similarity"])
    
    if avg_similarities:
        overall_avg_similarity = sum(avg_similarities) / len(avg_similarities)
        print(f"\n=== 批次攻擊結果 ===")
        print(f"處理批次數: {len(batch_results)}")
        print(f"處理文本總數: {total_processed}")
        print(f"成功處理: {total_successful}")
        print(f"整體平均相似度: {overall_avg_similarity:.3f}")
        
        # 系統統計
        stats = controller.get_statistics()
        print(f"\n=== 系統統計 ===")
        print(f"總運行次數: {stats['total_runs']}")
        print(f"平均處理時間: {stats['avg_processing_time']:.2f}秒")
        print(f"成功率: {stats['success_rate']:.1%}")
    
    # 保存結果
    if args.output:
        output_data = {
            "config": config.__dict__,
            "dataset_info": dataset,
            "batch_results": batch_results,
            "statistics": controller.get_statistics()
        }
        
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        print(f"\n結果已保存到: {args.output}")

def run_comprehensive_test(args) -> None:
    """執行綜合測試"""
    print("=== DART綜合測試模式 ===")
    
    config = DARTConfig(
        csv_path=args.csv_path,
        embedding_dim=args.embedding_dim,
        proximity_threshold=args.proximity_threshold,
        noise_std=args.noise_std,
        verbose=args.verbose
    )
    
    controller = DARTController(config)
    
    # 執行綜合測試
    test_results = controller.run_comprehensive_test(args.sample_size)
    
    if "error" in test_results:
        print(f"測試失敗: {test_results['error']}")
        return
    
    # 顯示測試結果
    test_config = test_results["test_config"]
    evaluation = test_results["evaluation"]
    
    print(f"\n=== 測試配置 ===")
    print(f"測試樣本數: {test_config['sample_size']}")
    print(f"有害文本: {test_config['harmful_texts']}")
    print(f"良性文本: {test_config['benign_texts']}")
    
    print(f"\n=== 測試結果 ===")
    for key, value in evaluation.items():
        if isinstance(value, float):
            print(f"{key}: {value:.3f}")
    
    # 顯示建議
    recommendations = test_results["recommendations"]
    if recommendations:
        print(f"\n=== 系統建議 ===")
        for i, rec in enumerate(recommendations, 1):
            print(f"{i}. {rec}")
    
    # 保存詳細結果
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(test_results, f, ensure_ascii=False, indent=2)
        print(f"\n詳細結果已保存到: {args.output}")

def main():
    """主函數"""
    parser = argparse.ArgumentParser(
        description="DART - 中文文本擾動攻擊系統",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 單次攻擊
  python main.py --mode single --texts "如何利用偏見操控他人？"
  
  # 批次攻擊
  python main.py --mode batch --csv-path problem.csv --sample-size 100
  
  # 綜合測試
  python main.py --mode test --sample-size 50 --output results.json
        """
    )
    
    # 主要選項
    parser.add_argument('--mode', choices=['single', 'batch', 'test'], 
                       required=True, help='運行模式')
    
    # 數據選項
    parser.add_argument('--csv-path', default='problem.csv',
                       help='CSV數據文件路徑')
    parser.add_argument('--texts', nargs='+', 
                       help='單次攻擊的文本列表')
    parser.add_argument('--sample-size', type=int, default=50,
                       help='樣本數量限制')
    
    # 模型參數
    parser.add_argument('--embedding-dim', type=int, default=512,
                       help='嵌入向量維度')
    parser.add_argument('--proximity-threshold', type=float, default=2.0,
                       help='鄰近性約束閾值')
    parser.add_argument('--noise-std', type=float, default=0.1,
                       help='噪聲標準差')
    
    # 批次選項
    parser.add_argument('--batch-size', type=int, default=8,
                       help='批次大小')
    
    # 輸出選項
    parser.add_argument('--output', help='結果輸出文件路徑')
    parser.add_argument('--verbose', action='store_true',
                       help='詳細輸出')
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       default='INFO', help='日誌級別')
    
    args = parser.parse_args()
    
    # 設置日誌
    setup_logging(args.log_level)
    
    # 驗證參數
    if args.mode == 'single' and not args.texts:
        parser.error("single模式需要提供--texts參數")
    
    if args.mode in ['batch', 'test'] and not Path(args.csv_path).exists():
        print(f"警告: CSV文件不存在: {args.csv_path}")
        print("將使用內置的測試數據")
    
    # 執行對應模式
    try:
        if args.mode == 'single':
            run_single_attack(args)
        elif args.mode == 'batch':
            run_batch_attack(args)
        elif args.mode == 'test':
            run_comprehensive_test(args)
    
    except KeyboardInterrupt:
        print("\n\n用戶中斷操作")
    except Exception as e:
        logging.error(f"程序執行錯誤: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()