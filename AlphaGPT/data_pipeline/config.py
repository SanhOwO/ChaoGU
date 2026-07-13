import os
from dotenv import load_dotenv

load_dotenv()

class AShareConfig:
    """A 股数据管线配置"""
    
    # 数据库（SQLite，简化部署）
    DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ashare_data.db")
    
    # AKShare 数据源
    DATA_SOURCE = "akshare"
    
    # A 股筛选条件
    MIN_HISTORY_DAYS = 60          # 最少上市 60 天
    MIN_PRICE = 2.0                # 最低股价 2 元
    MAX_PRICE = 500.0              # 最高股价 500 元
    MIN_MARKET_CAP = 1e8           # 最小流通市值 1 亿
    EXCLUDE_ST = True              # 排除 ST 股票
    EXCLUDE_NEW = True             # 排除上市 60 天内新股
    
    # 数据更新
    HISTORY_YEARS = 3              # 首次运行时下载 3 年历史
    UPDATE_MODE = "incremental"    # 增量更新
    
    # 并发
    CONCURRENCY = 10
    
    # 数据表名
    TABLE_DAILY = "daily_kline"
    TABLE_ADJUST = "adjust_factor"  # 复权因子（预留）
