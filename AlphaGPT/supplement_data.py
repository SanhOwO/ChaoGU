import sqlite3
import pandas as pd
from loguru import logger
from pathlib import Path

try:
    import akshare as ak
except ImportError:
    logger.error("akshare not installed. Run: pip install akshare")
    raise


def supplement_missing_data(db_path):
    """
    补充 SQLite 数据库中缺失的 turnover 和 market_cap 数据
    数据来源：AKShare 实时行情（一次性获取）
    """
    db_path = Path(db_path)
    if not db_path.exists():
        logger.error(f"Database not found: {db_path}")
        return
    
    conn = sqlite3.connect(db_path)
    
    # 1. 获取数据库中的股票代码列表
    codes_df = pd.read_sql("SELECT DISTINCT code FROM daily_kline", conn)
    db_codes = set(codes_df['code'].astype(str).str.strip().tolist())
    logger.info(f"Stocks in DB: {len(db_codes)}")
    
    # 2. 从 AKShare 获取全市场实时数据（含市值、换手率）
    logger.info("Fetching real-time market data from AKShare...")
    
    max_retries = 3
    spot_df = None
    for attempt in range(max_retries):
        try:
            spot_df = ak.stock_zh_a_spot_em()
            break
        except Exception as e:
            logger.warning(f"Attempt {attempt+1}/{max_retries} failed: {e}")
            import time
            time.sleep(3 * (attempt + 1))
    
    if spot_df is None or spot_df.empty:
        logger.error("Failed to fetch market data from AKShare")
        conn.close()
        return
    
    logger.success(f"Fetched {len(spot_df)} stocks from AKShare")
    
    # 标准化列名
    spot_df.columns = [c.strip() for c in spot_df.columns]
    spot_df['代码'] = spot_df['代码'].astype(str).str.strip()
    
    # 3. 提取需要的列
    supplement = pd.DataFrame({
        'code': spot_df['代码'],
        'turnover': pd.to_numeric(spot_df.get('换手率'), errors='coerce').fillna(0),
        'market_cap': pd.to_numeric(spot_df.get('流通市值'), errors='coerce').fillna(0),
        'close': pd.to_numeric(spot_df.get('最新价'), errors='coerce').fillna(0),
    })
    
    # 过滤：只保留数据库中已有的股票
    supplement = supplement[supplement['code'].isin(db_codes)]
    logger.info(f"Matched {len(supplement)} stocks with DB")
    
    # 4. 更新数据库
    cursor = conn.cursor()
    updated = 0
    
    for _, row in supplement.iterrows():
        code = row['code']
        turnover = float(row['turnover']) if pd.notna(row['turnover']) else 0.0
        market_cap = float(row['market_cap']) if pd.notna(row['market_cap']) else 0.0
        
        # 更新该股票所有记录的 turnover 和 circulating_market_cap
        # 由于市值和换手率会变化，我们用最新值作为近似
        cursor.execute(
            "UPDATE daily_kline SET turnover = ?, circulating_market_cap = ? WHERE code = ?",
            (turnover, market_cap, code)
        )
        updated += cursor.rowcount
    
    conn.commit()
    conn.close()
    
    logger.success(f"Updated {updated} records with turnover and market_cap")
    
    # 5. 统计
    conn = sqlite3.connect(db_path)
    stats = pd.read_sql("""
        SELECT 
            COUNT(*) as total,
            SUM(CASE WHEN turnover > 0 THEN 1 ELSE 0 END) as has_turnover,
            SUM(CASE WHEN circulating_market_cap > 0 THEN 1 ELSE 0 END) as has_mc
        FROM daily_kline
    """, conn)
    conn.close()
    
    logger.info(f"Database stats:")
    logger.info(f"  Total records: {stats['total'].iloc[0]}")
    logger.info(f"  With turnover: {stats['has_turnover'].iloc[0]} ({stats['has_turnover'].iloc[0]/stats['total'].iloc[0]*100:.1f}%)")
    logger.info(f"  With market_cap: {stats['has_mc'].iloc[0]} ({stats['has_mc'].iloc[0]/stats['total'].iloc[0]*100:.1f}%)")


if __name__ == "__main__":
    db_file = Path(__file__).parent / "data_pipeline" / "ashare_data.db"
    supplement_missing_data(db_file)
