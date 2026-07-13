"""
AlphaGPT-AShare 每日运行脚本

T 日 15:00 收盘后运行：
  1. 数据同步：AKShare → SQLite
  2. 信号生成：AI 公式打分
  3. 输出报告：T+1 日操作参考

使用方式：
  python daily_runner.py [--mode train|signal|sync]
"""

import argparse
import sys
from pathlib import Path

# 将项目根目录加入路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from loguru import logger

from data_pipeline.data_manager import AShareDataManager
from strategy_manager.signal_generator import AShareSignalGenerator
from model_core.engine import AShareAlphaEngine


def setup_logger():
    """配置日志"""
    logger.remove()
    logger.add(
        "daily_runner.log",
        rotation="1 day",
        retention="7 days",
        encoding="utf-8",
        level="INFO"
    )
    logger.add(sys.stdout, level="INFO")


def sync_data():
    """同步每日数据"""
    logger.info("=" * 60)
    logger.info("Step 1: Data Sync")
    logger.info("=" * 60)
    
    manager = AShareDataManager()
    manager.pipeline_sync_daily()
    logger.success("Data sync completed!")


def train_model():
    """训练模型"""
    logger.info("=" * 60)
    logger.info("Step 2: Model Training")
    logger.info("=" * 60)
    
    engine = AShareAlphaEngine(use_lord_regularization=True)
    engine.train()
    logger.success("Training completed!")


def generate_signals():
    """生成每日信号"""
    logger.info("=" * 60)
    logger.info("Step 3: Signal Generation")
    logger.info("=" * 60)
    
    generator = AShareSignalGenerator()
    report_path, df = generator.generate_daily_report()
    
    if report_path:
        logger.success(f"Report saved: {report_path}")
        
        # 打印 Top 5 信号
        print("\n" + "=" * 60)
        print("Top 5 Buy Signals:")
        print("=" * 60)
        top5 = df[df['score'] >= 0.80].head(5)
        if not top5.empty:
            for i, row in top5.iterrows():
                print(f"  {row['code']:>8} | Score: {row['score']:.4f} | Close: {row['close']:>8.2f} | {row['suggestion']}")
        else:
            print("  No strong signals today.")
        print("=" * 60)
    
    return report_path


def full_pipeline():
    """完整流程：同步 → 训练 → 信号（首次部署）"""
    setup_logger()
    
    print("\n" + "🚀" * 30)
    print("  AlphaGPT-AShare 全量运行")
    print("🚀" * 30 + "\n")
    
    # 1. 数据同步
    sync_data()
    
    # 2. 训练模型（如果已有最优公式可跳过）
    formula_path = Path("best_meme_strategy.json")
    if not formula_path.exists():
        logger.info("No trained formula found. Starting training...")
        train_model()
    else:
        logger.info("Trained formula found. Skipping training.")
    
    # 3. 生成信号
    generate_signals()
    
    print("\n" + "✅" * 30)
    print("  All tasks completed! Check daily_report_*.txt")
    print("✅" * 30 + "\n")


def daily_signal_only():
    """仅生成信号（日常模式，数据已同步）"""
    setup_logger()
    
    print("\n" + "📊" * 30)
    print("  AlphaGPT-AShare 每日信号生成")
    print("📊" * 30 + "\n")
    
    # 检查数据是否最新
    generator = AShareSignalGenerator()
    
    # 先生成信号，如果数据缺失再提醒同步
    try:
        report_path, df = generator.generate_daily_report()
        if report_path:
            print(f"\nReport: {report_path}")
    except Exception as e:
        logger.error(f"Failed to generate signals: {e}")
        logger.info("Please run: python daily_runner.py --mode sync")


def main():
    parser = argparse.ArgumentParser(description="AlphaGPT-AShare Daily Runner")
    parser.add_argument(
        "--mode", 
        choices=["full", "sync", "train", "signal"],
        default="signal",
        help="运行模式: full=完整流程, sync=仅同步数据, train=仅训练, signal=仅生成信号"
    )
    args = parser.parse_args()
    
    if args.mode == "full":
        full_pipeline()
    elif args.mode == "sync":
        setup_logger()
        sync_data()
    elif args.mode == "train":
        setup_logger()
        train_model()
    elif args.mode == "signal":
        daily_signal_only()


if __name__ == "__main__":
    main()
