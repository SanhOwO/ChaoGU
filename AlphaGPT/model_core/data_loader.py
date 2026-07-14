import sqlite3
import pandas as pd
import torch
from .config import ModelConfig
from .factors import AShareFactorEngineer

class AShareDataLoader:
    """A 股数据加载器：从 SQLite 读取，转为 Tensor"""
    
    def __init__(self):
        self.db_path = ModelConfig.DB_PATH
        self.feat_tensor = None      # [Stocks, Features, Time]
        self.raw_data_cache = None   # dict of tensors
        self.target_ret = None       # [Stocks, Time]
        self.codes = []              # 股票代码列表
        self.dates = []              # 交易日列表
        
    def load_data(self, limit_stocks=None, min_history=60):
        """
        从 SQLite 加载数据，构建特征张量
        
        Args:
            limit_stocks: 限制股票数量（None=全部）
            min_history: 最少历史天数（过滤新股）
        """
        print(f"Loading A-share data from {self.db_path}...")
        
        conn = sqlite3.connect(self.db_path)
        
        # 1. 获取最新交易日的全市场股票列表
        latest_date = pd.read_sql(
            "SELECT MAX(date) as max_date FROM daily_kline", conn
        )['max_date'].iloc[0]
        
        # 2. 获取候选股票
        codes_query = f"""
        SELECT code, COUNT(*) as cnt 
        FROM daily_kline 
        GROUP BY code 
        HAVING cnt >= {min_history}
        ORDER BY cnt DESC
        """
        if limit_stocks:
            codes_query += f" LIMIT {limit_stocks}"
        
        codes_df = pd.read_sql(codes_query, conn)
        self.codes = codes_df['code'].tolist()
        
        if not self.codes:
            raise ValueError("No stocks found in database. Run data pipeline first.")
        
        print(f"Selected {len(self.codes)} stocks with >= {min_history} days history.")
        
        # 3. 构建统一时间轴（交易日序列）
        dates_query = """
        SELECT DISTINCT date FROM daily_kline 
        WHERE code = (SELECT code FROM daily_kline LIMIT 1)
        ORDER BY date
        """
        self.dates = pd.read_sql(dates_query, conn)['date'].tolist()
        
        # 4. 对每个股票取数据并 pivot
        code_str = "','".join(self.codes)
        data_query = f"""
        SELECT date, code, open, high, low, close, volume, amount, 
               turnover, market_cap
        FROM daily_kline
        WHERE code IN ('{code_str}')
        ORDER BY date ASC
        """
        df = pd.read_sql(data_query, conn)
        conn.close()
        
        # 5. 构建 tensor 矩阵 [Stocks, Time]
        def to_tensor(col):
            pivot = df.pivot(index='date', columns='code', values=col)
            pivot = pivot.reindex(columns=self.codes)
            pivot = pivot.ffill().fillna(0.0)
            return torch.tensor(pivot.values.T, dtype=torch.float32, device=ModelConfig.DEVICE)
        
        self.raw_data_cache = {
            'open': to_tensor('open'),
            'high': to_tensor('high'),
            'low': to_tensor('low'),
            'close': to_tensor('close'),
            'volume': to_tensor('volume'),
            'amount': to_tensor('amount'),
            'turnover': to_tensor('turnover'),
            'market_cap': to_tensor('market_cap'),
        }
        
        # 6. 计算特征张量
        self.feat_tensor = AShareFactorEngineer.compute_features(self.raw_data_cache)
        
        # 7. 计算目标收益：T+1 开盘买入，T+2 开盘卖出
        # target_ret[t] = open[t+2] / open[t+1] - 1
        op = self.raw_data_cache['open']
        vol = self.raw_data_cache['volume']
        
        # 向量化屏蔽停牌日：T+1 或 T+2 是停牌日（volume=0）则无法完整交易
        t1 = torch.roll(op, -1, dims=1)   # T+1 开盘
        t2 = torch.roll(op, -2, dims=1)   # T+2 开盘
        vol_t1 = torch.roll(vol, -1, dims=1)   # T+1 成交量
        vol_t2 = torch.roll(vol, -2, dims=1)   # T+2 成交量
        
        # 只有 T+1 和 T+2 都正常交易的日子才计算收益率
        can_trade = (vol_t1 > 0) & (vol_t2 > 0) & (t1 > 0) & (t2 > 0)
        
        self.target_ret = torch.where(
            can_trade,
            (t2 / (t1 + 1e-9)) - 1.0,
            0.0
        )
        
        # 最后两期无法计算目标收益，置零
        self.target_ret[:, -2:] = 0.0
        
        # 额外保护：截断极端异常值
        self.target_ret = torch.clamp(self.target_ret, -0.5, 0.5)

        
        print(f"Data Ready. Shape: {self.feat_tensor.shape}")
        print(f"  Stocks: {self.feat_tensor.shape[0]}")
        print(f"  Features: {self.feat_tensor.shape[1]}")
        print(f"  TimeSteps: {self.feat_tensor.shape[2]}")
        
    def get_latest_snapshot(self, n_days=60):
        """
        获取最近 n 天的数据快照，用于每日信号生成
        Returns: (feat_tensor_slice, codes, latest_date)
        """
        if self.feat_tensor is None:
            self.load_data()
        
        T = self.feat_tensor.shape[2]
        start = max(0, T - n_days)
        
        feat_slice = self.feat_tensor[:, :, start:]
        latest_date = self.dates[-1] if self.dates else "Unknown"
        
        return feat_slice, self.codes, latest_date
    
    def get_stock_info(self, code):
        """获取单只股票的最新信息"""
        if code not in self.codes:
            return None
        idx = self.codes.index(code)
        return {
            'code': code,
            'latest_close': self.raw_data_cache['close'][idx, -1].item(),
            'latest_open': self.raw_data_cache['open'][idx, -1].item(),
            'latest_volume': self.raw_data_cache['volume'][idx, -1].item(),
            'market_cap': self.raw_data_cache['market_cap'][idx, -1].item(),
        }
