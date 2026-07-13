import time
import akshare as ak
import pandas as pd
from datetime import datetime, timedelta
from loguru import logger

class AKShareProvider:
    """AKShare A 股数据提供者（含重试机制）"""
    
    def __init__(self):
        self.source = "akshare"
    
    def _retry_call(self, func, max_retries=3, sleep_sec=2, *args, **kwargs):
        """带重试的 AKShare 调用"""
        for attempt in range(max_retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.warning(f"Attempt {attempt+1}/{max_retries} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(sleep_sec * (attempt + 1))
                else:
                    raise
        return None
    
    def get_all_stock_list(self):
        """
        获取全市场 A 股列表（排除北交所、B股）
        Returns: DataFrame[code, name, industry, list_date]
        """
        # 主接口：stock_zh_a_spot_em（实时行情，包含市值）
        try:
            df = self._retry_call(ak.stock_zh_a_spot_em, max_retries=3, sleep_sec=3)
            if df is not None and not df.empty:
                df.columns = [c.strip() for c in df.columns]
                result = pd.DataFrame({
                    'code': df['代码'].astype(str).str.strip(),
                    'name': df['名称'],
                    'price': pd.to_numeric(df['最新价'], errors='coerce'),
                    'change_pct': pd.to_numeric(df['涨跌幅'], errors='coerce'),
                    'volume': pd.to_numeric(df['成交量'], errors='coerce'),
                    'amount': pd.to_numeric(df['成交额'], errors='coerce'),
                    'turnover': pd.to_numeric(df['换手率'], errors='coerce'),
                    'market_cap': pd.to_numeric(df['流通市值'], errors='coerce'),
                    'total_cap': pd.to_numeric(df['总市值'], errors='coerce'),
                    'pe_ttm': pd.to_numeric(df['市盈率-动态'], errors='coerce'),
                    'pb': pd.to_numeric(df['市净率'], errors='coerce'),
                })
                result = result[~result['code'].str.startswith(('8', '4', '9'))]
                result = result[~result['code'].str.endswith(('B', 'b'))]
                logger.success(f"Fetched {len(result)} stocks via stock_zh_a_spot_em")
                return result
        except Exception as e:
            logger.warning(f"Primary stock list API failed: {e}")
        
        # 备用接口：stock_info_a_code_name（基础列表，不含市值）
        try:
            logger.info("Trying backup API: stock_info_a_code_name")
            df = self._retry_call(ak.stock_info_a_code_name, max_retries=3, sleep_sec=2)
            if df is not None and not df.empty:
                result = pd.DataFrame({
                    'code': df['code'].astype(str).str.strip(),
                    'name': df['name'],
                    'price': 0.0,
                    'change_pct': 0.0,
                    'volume': 0.0,
                    'amount': 0.0,
                    'turnover': 0.0,
                    'market_cap': 0.0,
                    'total_cap': 0.0,
                    'pe_ttm': 0.0,
                    'pb': 0.0,
                })
                result = result[~result['code'].str.startswith(('8', '4', '9'))]
                result = result[~result['code'].str.endswith(('B', 'b'))]
                logger.success(f"Fetched {len(result)} stocks via backup API")
                return result
        except Exception as e:
            logger.error(f"Backup API also failed: {e}")
        
        return pd.DataFrame()
    
    def get_stock_history(self, code, start_date=None, end_date=None):
        """
        获取单只股票历史日 K 数据（带重试）
        """
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=365*3)).strftime('%Y%m%d')
        if end_date is None:
            end_date = datetime.now().strftime('%Y%m%d')
        
        try:
            df = self._retry_call(
                ak.stock_zh_a_hist,
                max_retries=3, sleep_sec=2,
                symbol=code, period="daily",
                start_date=start_date, end_date=end_date,
                adjust="qfq"
            )
            
            if df is None or df.empty:
                return pd.DataFrame()
            
            df = df.rename(columns={
                '日期': 'date', '开盘': 'open', '收盘': 'close',
                '最高': 'high', '最低': 'low', '成交量': 'volume',
                '成交额': 'amount', '振幅': 'amplitude',
                '涨跌幅': 'change_pct', '涨跌额': 'change_amount',
                '换手率': 'turnover',
            })
            
            df['code'] = code
            df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
            df['circulating_market_cap'] = 0.0
            
            return df[['date', 'code', 'open', 'high', 'low', 'close', 
                       'volume', 'amount', 'turnover', 'circulating_market_cap']]
            
        except Exception as e:
            logger.warning(f"Failed to fetch history for {code}: {e}")
            return pd.DataFrame()
    
    def get_stock_capital_data(self, code):
        """获取股本数据（带重试）"""
        try:
            df = self._retry_call(ak.stock_individual_info_em, max_retries=2, sleep_sec=1, symbol=code)
            if df is None or df.empty:
                return {}
            info = dict(zip(df['item'], df['value']))
            return {
                'total_shares': float(info.get('总股本', 0)),
                'circulating_shares': float(info.get('流通股本', 0)),
            }
        except Exception as e:
            logger.warning(f"Failed to fetch capital for {code}: {e}")
            return {}
