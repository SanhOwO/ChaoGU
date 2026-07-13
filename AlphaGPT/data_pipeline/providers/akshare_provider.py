import akshare as ak
import pandas as pd
from datetime import datetime, timedelta
from loguru import logger

class AKShareProvider:
    """AKShare A 股数据提供者"""
    
    def __init__(self):
        self.source = "akshare"
    
    def get_all_stock_list(self):
        """
        获取全市场 A 股列表（排除北交所、B股）
        Returns: DataFrame[code, name, industry, list_date]
        """
        try:
            df = ak.stock_zh_a_spot_em()
            # 清洗列名
            df.columns = [c.strip() for c in df.columns]
            
            # 选取需要的列
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
            
            # 过滤：排除北交所（代码 8 开头）、B股、新三板
            result = result[~result['code'].str.startswith(('8', '4', '9'))]
            result = result[~result['code'].str.endswith(('B', 'b'))]
            
            return result
        except Exception as e:
            logger.error(f"Failed to fetch stock list: {e}")
            return pd.DataFrame()
    
    def get_stock_history(self, code, start_date=None, end_date=None):
        """
        获取单只股票历史日 K 数据
        
        Args:
            code: 股票代码（如 '000001'）
            start_date: 开始日期 'YYYYMMDD'
            end_date: 结束日期 'YYYYMMDD'
        """
        try:
            if start_date is None:
                start_date = (datetime.now() - timedelta(days=365*3)).strftime('%Y%m%d')
            if end_date is None:
                end_date = datetime.now().strftime('%Y%m%d')
            
            # AKShare 接口：日 K 数据
            df = ak.stock_zh_a_hist(
                symbol=code,
                period="daily",
                start_date=start_date,
                end_date=end_date,
                adjust="qfq"  # 前复权
            )
            
            if df.empty:
                return pd.DataFrame()
            
            # 标准化列名
            df = df.rename(columns={
                '日期': 'date',
                '开盘': 'open',
                '收盘': 'close',
                '最高': 'high',
                '最低': 'low',
                '成交量': 'volume',
                '成交额': 'amount',
                '振幅': 'amplitude',
                '涨跌幅': 'change_pct',
                '涨跌额': 'change_amount',
                '换手率': 'turnover',
            })
            
            df['code'] = code
            df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
            
            # 计算流通市值（需要额外接口，这里用最新市值近似）
            # 实际应用中应该从财务数据接口获取
            df['circulating_market_cap'] = 0.0  # 后续通过 stock_list 映射填充
            
            return df[['date', 'code', 'open', 'high', 'low', 'close', 
                       'volume', 'amount', 'turnover', 'circulating_market_cap']]
            
        except Exception as e:
            logger.warning(f"Failed to fetch history for {code}: {e}")
            return pd.DataFrame()
    
    def get_stock_capital_data(self, code):
        """
        获取股本数据（用于计算流通市值）
        """
        try:
            df = ak.stock_individual_info_em(symbol=code)
            if df.empty:
                return {}
            info = dict(zip(df['item'], df['value']))
            return {
                'total_shares': float(info.get('总股本', 0)),
                'circulating_shares': float(info.get('流通股本', 0)),
            }
        except Exception as e:
            logger.warning(f"Failed to fetch capital for {code}: {e}")
            return {}
