import os
import pandas as pd
import sqlite3
from tqdm import tqdm
import re

DATA_DIR = r"G:\学习理财赚钱\量化\AI\AlphaGPT个股策略\AlphaGPT\数据\csi1000_hist_data"
DB_PATH = r"G:\学习理财赚钱\量化\AI\AlphaGPT个股策略\AlphaGPT\data_pipeline\ashare_data.db"

def import_csi1000():
    csv_files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith('.csv')])
    print(f"Found {len(csv_files)} CSV files")

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    # Drop old table and create new one
    cur.execute("DROP TABLE IF EXISTS daily_bars")
    cur.execute("""
        CREATE TABLE daily_bars (
            date TEXT,
            code TEXT,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume REAL,
            amount REAL,
            turnover REAL,
            market_cap REAL
        )
    """)
    conn.commit()

    total_rows = 0
    skipped = []
    for fname in tqdm(csv_files, desc="Importing"):
        if not fname.endswith('.csv'):
            continue
        fpath = os.path.join(DATA_DIR, fname)
        try:
            df = pd.read_csv(fpath, encoding='utf-8-sig')
        except Exception as e:
            skipped.append((fname, str(e)))
            continue

        # Rename columns
        col_map = {
            '日期': 'date',
            '股票代码': 'code',
            '开盘': 'open',
            '收盘': 'close',
            '最高': 'high',
            '最低': 'low',
            '成交量': 'volume',
            '成交额': 'amount',
            '换手率': 'turnover',
            '流通市值': 'market_cap',
        }
        # Only rename columns that exist
        df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

        if 'code' not in df.columns:
            skipped.append((fname, "missing 'code' column"))
            continue

        # Clean code: remove 'sz.' or 'sh.' prefix
        df['code'] = df['code'].astype(str).str.replace(r'^(sz|sh)\.', '', regex=True)

        # Select needed columns (only those that exist)
        needed = ['date', 'code', 'open', 'high', 'low', 'close', 'volume', 'amount', 'turnover', 'market_cap']
        available = [c for c in needed if c in df.columns]
        df = df[available]

        df.to_sql('daily_bars', conn, if_exists='append', index=False)
        total_rows += len(df)
    for fname in tqdm(csv_files, desc="Importing"):
        fpath = os.path.join(DATA_DIR, fname)
        df = pd.read_csv(fpath, encoding='utf-8-sig')

        # Rename columns
        col_map = {
            '日期': 'date',
            '股票代码': 'code',
            '开盘': 'open',
            '收盘': 'close',
            '最高': 'high',
            '最低': 'low',
            '成交量': 'volume',
            '成交额': 'amount',
            '换手率': 'turnover',
            '流通市值': 'market_cap',
        }
        df = df.rename(columns=col_map)

        # Clean code: remove 'sz.' or 'sh.' prefix
        df['code'] = df['code'].str.replace(r'^(sz|sh)\.', '', regex=True)

        # Select needed columns
        df = df[['date', 'code', 'open', 'high', 'low', 'close', 'volume', 'amount', 'turnover', 'market_cap']]

        df.to_sql('daily_bars', conn, if_exists='append', index=False)
        total_rows += len(df)

    conn.commit()

    # Verify
    cur.execute("SELECT COUNT(*) FROM daily_bars")
    total = cur.fetchone()[0]
    cur.execute("SELECT COUNT(DISTINCT code) FROM daily_bars")
    stocks = cur.fetchone()[0]
    cur.execute("SELECT MIN(date), MAX(date) FROM daily_bars")
    date_range = cur.fetchone()
    cur.execute("SELECT COUNT(*) FROM daily_bars WHERE turnover IS NOT NULL")
    has_turnover = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM daily_bars WHERE market_cap IS NOT NULL")
    has_cap = cur.fetchone()[0]

    conn.close()

    print(f"\n=== Import Complete ===")
    print(f"Total records: {total:,}")
    print(f"Distinct stocks: {stocks}")
    print(f"Date range: {date_range[0]} ~ {date_range[1]}")
    print(f"Records with turnover: {has_turnover:,} ({has_turnover/total*100:.1f}%)")
    print(f"Records with market_cap: {has_cap:,} ({has_cap/total*100:.1f}%)")

if __name__ == '__main__':
    import_csi1000()
