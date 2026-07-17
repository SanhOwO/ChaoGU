import sys
import os

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(project_root, 'AlphaGPT'))

import sqlite3
import pandas as pd
from glob import glob

def import_csv_to_db(csv_dir=None, db_path=None):
    """
    将 CSV 个股数据导入 SQLite 数据库。
    
    扫描 csv_dir 下的所有 .csv 文件，读取后写入 db_path 的 daily_kline 表。
    已存在的数据会被替换（先删除同 code 的旧数据，再插入新数据）。
    """
    if csv_dir is None:
        csv_dir = os.path.join(project_root, "AlphaGPT", "数据", "single_hist_data")
    if db_path is None:
        db_path = os.path.join(project_root, "AlphaGPT", "data_pipeline", "ashare_data.db")
    
    csv_files = glob(os.path.join(csv_dir, "*.csv"))
    
    if not csv_files:
        print(f"[WARN] No CSV files found in {csv_dir}")
        return
    
    print("=" * 60)
    print("CSV to SQLite Importer")
    print(f"Database: {db_path}")
    print(f"CSV files: {len(csv_files)}")
    print("=" * 60)
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # 确保表结构存在（兼容现有 daily_kline 表）
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
            market_cap REAL,
            PRIMARY KEY (date, code)
        )
    """)
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_code ON daily_kline (code)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_date ON daily_kline (date)")
    conn.commit()
    
    total_inserted = 0
    
    for csv_path in csv_files:
        fname = os.path.basename(csv_path)
        print(f"\nProcessing: {fname}")
        
        df = pd.read_csv(csv_path)
        
        if df.empty:
            print(f"  [SKIP] Empty file")
            continue
        
        # 列名映射（中文 CSV → 英文数据库）
        col_map = {
            '日期': 'date',
            '股票代码': 'code_raw',
            '开盘': 'open',
            '收盘': 'close',
            '最高': 'high',
            '最低': 'low',
            '成交量': 'volume',
            '成交额': 'amount',
            '换手率': 'turnover',
            '流通市值': 'market_cap',
        }
        
        # 检查必要列是否存在
        missing = [c for c in col_map.keys() if c not in df.columns]
        if missing:
            print(f"  [SKIP] Missing columns: {missing}")
            continue
        
        # 提取目标列
        df_out = df[list(col_map.keys())].copy()
        df_out.columns = list(col_map.values())
        
        # 处理股票代码：sz.002548 → 002548
        df_out['code'] = df_out['code_raw'].astype(str).str.replace(r'^[a-zA-Z]+\.', '', regex=True)
        df_out.drop(columns=['code_raw'], inplace=True)
        
        # 确保日期格式统一
        df_out['date'] = pd.to_datetime(df_out['date']).dt.strftime('%Y-%m-%d')
        
        # 删除该股票在数据库中的旧数据（防止重复）
        stock_code = df_out['code'].iloc[0]
        cursor.execute("DELETE FROM daily_kline WHERE code = ?", (stock_code,))
        conn.commit()
        
        # 分批插入，每批 500 行（避免 SQLite 变量限制）
        chunk_size = 500
        for start in range(0, len(df_out), chunk_size):
            chunk = df_out.iloc[start:start+chunk_size]
            chunk.to_sql('daily_kline', conn, if_exists='append', index=False)
        
        inserted = len(df_out)
        total_inserted += inserted
        print(f"  [OK] Inserted {inserted} rows | Code: {stock_code} | Date: {df_out['date'].min()} ~ {df_out['date'].max()}")
    
    conn.close()
    
    print(f"\n{'=' * 60}")
    print(f"Import complete! Total inserted: {total_inserted} rows")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    import_csv_to_db()
