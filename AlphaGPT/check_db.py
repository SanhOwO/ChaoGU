import sqlite3, sys

db_path = r"G:\学习理财赚钱\量化\AI\AlphaGPT个股策略\AlphaGPT\data_pipeline\ashare_data.db"
conn = sqlite3.connect(db_path)
cur = conn.cursor()

print("=== 数据库表 ===")
cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
print([r[0] for r in cur.fetchall()])

print("\n=== daily_bars 列 ===")
cur.execute("PRAGMA table_info(daily_bars)")
cols = [r[1] for r in cur.fetchall()]
print(cols)

print("\n=== 数据概览 ===")
cur.execute("SELECT COUNT(*) FROM daily_bars")
total = cur.fetchone()[0]
print(f"Total records: {total:,}")

cur.execute("SELECT COUNT(DISTINCT code) FROM daily_bars")
stocks = cur.fetchone()[0]
print(f"Distinct stocks: {stocks}")

cur.execute("SELECT COUNT(*) FROM daily_bars WHERE turnover IS NOT NULL")
has_turnover = cur.fetchone()[0]
print(f"Records with turnover: {has_turnover:,} ({has_turnover/total*100:.1f}%)")

cur.execute("SELECT COUNT(DISTINCT code) FROM daily_bars WHERE turnover IS NOT NULL")
turnover_stocks = cur.fetchone()[0]
print(f"Stocks with turnover: {turnover_stocks} ({turnover_stocks/stocks*100:.1f}%)")

cur.execute("SELECT COUNT(*) FROM daily_bars WHERE market_cap IS NOT NULL")
has_cap = cur.fetchone()[0]
print(f"Records with market_cap: {has_cap:,} ({has_cap/total*100:.1f}%)")

cur.execute("SELECT COUNT(DISTINCT code) FROM daily_bars WHERE market_cap IS NOT NULL")
cap_stocks = cur.fetchone()[0]
print(f"Stocks with market_cap: {cap_stocks} ({cap_stocks/stocks*100:.1f}%)")

cur.execute("SELECT COUNT(*) FROM daily_bars WHERE amount IS NOT NULL")
has_amount = cur.fetchone()[0]
print(f"Records with amount: {has_amount:,} ({has_amount/total*100:.1f}%)")

cur.execute("SELECT COUNT(DISTINCT code) FROM daily_bars WHERE amount IS NOT NULL")
amount_stocks = cur.fetchone()[0]
print(f"Stocks with amount: {amount_stocks} ({amount_stocks/stocks*100:.1f}%)")

print("\n=== 数据日期范围 ===")
cur.execute("SELECT MIN(date), MAX(date) FROM daily_bars")
print(cur.fetchone())

print("\n=== 样本数据（最近一条） ===")
cur.execute("SELECT code, date, close, volume, amount, turnover, market_cap FROM daily_bars ORDER BY date DESC LIMIT 1")
print(cur.fetchone())

conn.close()
print("\n=== Done ===")
