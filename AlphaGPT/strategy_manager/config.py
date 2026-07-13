class AShareStrategyConfig:
    """A 股策略配置（人工操作版）"""
    
    # 仓位管理
    MAX_OPEN_POSITIONS = 5         # 最多同时持有 5 只股票
    ENTRY_AMOUNT_PER_STOCK = 10000  # 每只股票投入 1 万元
    MAX_TOTAL_POSITION = 50000      # 总仓位上限 5 万元
    
    # 信号阈值
    BUY_THRESHOLD = 0.80            # 买入信号阈值（分数 >= 0.80 考虑买入）
    SELL_THRESHOLD = 0.35           # 卖出信号阈值（分数 <= 0.35 考虑卖出）
    TOP_N_STOCKS = 20               # 每日关注 Top 20
    
    # 风控（人工参考）
    STOP_LOSS_PCT = -0.05           # 止损线 -5%
    TAKE_PROFIT_PCT = 0.08          # 止盈线 +8%
    TRAILING_ACTIVATION = 0.05      # 移动止盈激活条件：最大盈利 > 5%
    TRAILING_DROP = 0.03            # 移动止盈回撤：回撤 > 3%
    
    # 排除条件
    MIN_DAILY_AMOUNT = 1e6          # 最低日成交额 100 万（流动性过滤）
    MIN_SCORE_SPREAD = 0.05        # 最低分数间隔（Top 1 和 Top 2 差距）
