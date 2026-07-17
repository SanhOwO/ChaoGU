import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'AlphaGPT'))

from model_core.engine import AShareAlphaEngine
from model_core.config import ModelConfig

if __name__ == '__main__':
    # 用法：python train_single.py 300633
    # 或修改下方 default_code 后直接运行
    
    default_code = '300633'  # ← 默认训练的个股，按需修改
    
    if len(sys.argv) > 1:
        stock_code = sys.argv[1]
    else:
        stock_code = default_code
        print(f"[INFO] 未指定股票代码，使用默认: {stock_code}")
        print(f"       用法: python train_single.py <股票代码>")
    
    # 强制只训练单只个股
    ModelConfig.TRAIN_SPECIFIC_CODES = [stock_code]
    
    print("=" * 60)
    print(f"AlphaGPT A-Share Single Stock Training")
    print(f"Target Stock: {stock_code}")
    print("=" * 60)
    
    eng = AShareAlphaEngine(use_lord_regularization=True)
    eng.train(save_prefix=stock_code)
