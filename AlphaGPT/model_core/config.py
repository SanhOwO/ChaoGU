import os
import torch
from .vocab import FORMULA_VOCAB

class ModelConfig:
    """A 股模型配置"""
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 数据库改用 SQLite（个人量化，简化部署）
    DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                           "data_pipeline", "ashare_data.db")
    
    BATCH_SIZE = 4096        # A 股全市场约 5000+ 只股票，减小 batch
    TRAIN_STEPS = 2000       # 增加训练步数（A 股数据更丰富）
    MAX_FORMULA_LEN = 14     # 增加公式长度上限（A 股因子更多）
    
    # A 股交易参数（用于回测评分）
    TRADE_SIZE_CNY = 10000.0   # 每只股票假设投入 1 万元
    BASE_FEE_BUY = 0.00025     # 买入佣金 0.025%
    BASE_FEE_SELL = 0.00075    # 卖出佣金 0.025% + 印花税 0.05%
    MIN_COMMISSION = 5.0       # 最低佣金 5 元
    SLIPPAGE = 0.001           # 滑点 0.1%
    
    # 排除条件
    MIN_PRICE = 2.0            # 排除低价股（<2元）
    MAX_PRICE = 500.0          # 排除高价股（>500元，极少）
    MIN_MARKET_CAP = 1e8       # 最小流通市值 1亿
    
    INPUT_DIM = FORMULA_VOCAB.feature_count
