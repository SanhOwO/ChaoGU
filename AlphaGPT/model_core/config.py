import os
import torch
from .vocab import FORMULA_VOCAB

class ModelConfig:
    """A 股模型配置"""
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 数据库改用 SQLite（个人量化，简化部署）
    DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                           "data_pipeline", "ashare_data.db")
    
    BATCH_SIZE = 1024        # 减小batch以加快训练速度
    TRAIN_STEPS = 2000       # 增加训练步数（A 股数据更丰富）
    MAX_FORMULA_LEN = 14     # 增加公式长度上限（A 股因子更多）
    
    # A 股交易参数（真实交易成本：万一佣金 + 0.05%印花税 + 0.001%过户费）
    TRADE_SIZE_CNY = 10000.0   # 每只股票假设投入 1 万元
    BASE_FEE_BUY = 0.00011     # 买入成本：佣金 0.01% + 过户费 0.001% = 0.011%
    BASE_FEE_SELL = 0.00061    # 卖出成本：佣金 0.01% + 印花税 0.05% + 过户费 0.001% = 0.061%
    MIN_COMMISSION = 0.1       # 最低佣金 0.1 元
    SLIPPAGE = 0.001           # 滑点 0.1%（双边）
    
    # 完整买卖一次总成本 ≈ 0.011% + 0.061% + 0.1%*2 = 0.272%
    
    # 排除条件
    MIN_PRICE = 2.0            # 排除低价股（<2元）
    MAX_PRICE = 500.0          # 排除高价股（>500元，极少）
    MIN_MARKET_CAP = 1e8       # 最小流通市值 1亿
    
    INPUT_DIM = FORMULA_VOCAB.feature_count
