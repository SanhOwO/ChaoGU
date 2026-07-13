import sqlite3
import pandas as pd
from pathlib import Path
from loguru import logger


def import_csv_to_sqlite(csv_path, db_path, chunk_size=100000):
    """
    将 all_stock_cache.csv 导入 SQLite 数据库
    
    Args:
        csv_path: CSV 文件路径
        db_path: SQLite 数据库路径
        chunk_size: 每批读取行数（避免内存溢出）
    """
    csv_path = Path(csv_path)
    db_path = Path(db_path)
    
    if not csv_path.exists():
        logger.error(f"CSV not found: {csv_path}")
        return
    
    logger.info(f"Importing CSV: {csv_path} ({csv_path.stat().st_size / 1024 / 1024:.1f} MB)")
    
    # 初始化数据库
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS daily_kline (
            date TEXT NOT NULL,
            code TEXT NOT NULL,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume REAL,
            amount REAL,
            turnover REAL,
            circulating_market_cap REAL,
            PRIMARY KEY (date, code)
        )
    """)
    
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_code ON daily_kline (code)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_date ON daily_kline (date)")
    
    conn.commit()
    
    # 先清空旧数据（全量导入）
    cursor.execute("DELETE FROM daily_kline")
    conn.commit()
    
    # 逐块读取 CSV
    total_rows = 0
    for chunk in pd.read_csv(csv_path, chunksize=chunk_size):
        # 标准化列名
        chunk = chunk.rename(columns={
            'symbol': 'code',
        })
        
        # 补全前导零：1 → 000001, 600519 → 600519, 300001 → 300001
        chunk['code'] = chunk['code'].astype(str).str.strip()
        
        def pad_code(c):
            """根据代码规则补全前导零"""
            if len(c) == 6:
                return c  # 已经是 6 位
            elif len(c) < 6:
                # 沪市 6 开头，深市 0/3 开头
                if c.startswith('6') or c.startswith('68') or c.startswith('60') or c.startswith('69'):
                    return c.zfill(6)
                else:
                    return c.zfill(6)
            return c[:6]
        
        chunk['code'] = chunk['code'].apply(pad_code)
        
        # 计算 amount（成交额 = close * volume，单位：元）
        chunk['amount'] = chunk['close'] * chunk['volume']
        
        # 缺失的 turnover 和 market_cap 用 0 填充
        chunk['turnover'] = 0.0
        chunk['circulating_market_cap'] = 0.0
        
        # 标准化日期格式
        chunk['date'] = pd.to_datetime(chunk['date']).dt.strftime('%Y-%m-%d')
        
        # 选择需要的列，顺序与数据库一致
        chunk = chunk[['date', 'code', 'open', 'high', 'low', 'close', 
                       'volume', 'amount', 'turnover', 'circulating_market_cap']]
        
        # 使用 executemany 批量插入，避免 too many variables 错误
        chunk_tuples = chunk.values.tolist()
        cursor.executemany(
            "INSERT INTO daily_kline VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            chunk_tuples
        )
        conn.commit()
        total_rows += len(chunk)
        logger.info(f"Imported {total_rows} rows...")
    
    # 清理重复（由于 append 模式）
    cursor.execute("""
        DELETE FROM daily_kline
        WHERE rowid NOT IN (
            SELECT MIN(rowid)
            FROM daily_kline
            GROUP BY date, code
        )
    """)
    
    # 统计
    cursor.execute("SELECT COUNT(*) FROM daily_kline")
    total = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(DISTINCT code) FROM daily_kline")
    stocks = cursor.fetchone()[0]
    cursor.execute("SELECT MIN(date), MAX(date) FROM daily_kline")
    date_range = cursor.fetchone()
    
    conn.commit()
    conn.close()
    
    logger.success("Import complete!")
    logger.info(f"  Total records: {total}")
    logger.info(f"  Total stocks: {stocks}")
    logger.info(f"  Date range: {date_range[0]} ~ {date_range[1]}")
    logger.info(f"  Database: {db_path}")
    
    return db_path


if __name__ == "__main__":
    # 默认路径
    csv_file = Path(__file__).parent / "数据" / "all_stock_cache.csv"
    db_file = Path(__file__).parent / "data_pipeline" / "ashare_data.db"
    
    import_csv_to_sqlite(csv_file, db_file)
