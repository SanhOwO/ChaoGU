import sqlite3

db = r"G:/学习理财赚钱/量化/AI/AlphaGPT个股策略/AlphaGPT/data_pipeline/ashare_data.db"
conn = sqlite3.connect(db)
cur = conn.cursor()

# Drop old daily_kline, rename daily_bars to daily_kline
cur.execute("DROP TABLE IF EXISTS daily_kline")
cur.execute("ALTER TABLE daily_bars RENAME TO daily_kline")
conn.commit()

# Verify
cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
print("Tables:", [r[0] for r in cur.fetchall()])

cur.execute("SELECT COUNT(*) FROM daily_kline")
print(f"daily_kline: {cur.fetchone()[0]:,} records")

cur.execute("SELECT COUNT(DISTINCT code) FROM daily_kline")
print(f"Stocks: {cur.fetchone()[0]}")

conn.close()
print("Done!")
