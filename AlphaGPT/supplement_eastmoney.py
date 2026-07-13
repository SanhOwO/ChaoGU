import os
import time
import subprocess
import json
import pandas as pd
import sqlite3
from pathlib import Path
from loguru import logger

# 禁用代理干扰
os.environ['NO_PROXY'] = '*'
os.environ['no_proxy'] = '*'


def fetch_eastmoney_spot_curl(page=1, page_size=100):
    """
    用 curl 直接调用东方财富 API（绕过 Python requests 的代理问题）
    """
    url = (
        f"https://push2.eastmoney.com/api/qt/clist/get"
        f"?pn={page}&pz={page_size}&po=1&np=1&fltt=2&invt=2&fid=f12"
        f"&fs=m:0+t:6,m:0+t:80,m:1+t:2,m:1+t:23,m:0+t:81+s:2048"
        f"&fields=f12,f14,f2,f8,f21"
    )
    
    cmd = [
        'curl', '-s', '--max-time', '15', '--connect-timeout', '10',
        '-H', 'User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        '-H', 'Referer: https://quote.eastmoney.com/',
        url
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=20)
        if result.returncode != 0:
            logger.warning(f"Page {page}: curl failed ({result.returncode})")
            return None
        
        stdout_text = result.stdout.decode('utf-8', errors='ignore')
        data = json.loads(stdout_text)
        diffs = data.get('data', {}).get('diff', [])
        if not diffs:
            return None
        
        records = []
        for d in diffs:
            code = str(d.get('f12', '')).strip()
            if not code or code == 'None':
                continue
            
            # 处理非数字值（如 '-' 表示停牌或无数据）
            def safe_float(v):
                try:
                    return float(v) if v not in ('-', '', None, 'None') else 0.0
                except (ValueError, TypeError):
                    return 0.0
            
            records.append({
                'code': code,
                'name': str(d.get('f14', '')).strip(),
                'close': safe_float(d.get('f2')),
                'turnover': safe_float(d.get('f8')),
                'market_cap': safe_float(d.get('f21')),
            })
        
        return records
    except Exception as e:
        logger.warning(f"Page {page} fetch failed: {e}")
        return None


def supplement_from_eastmoney(db_path):
    """
    从东方财富 API 分页获取全市场数据，补充 turnover 和 market_cap
    """
    db_path = Path(db_path)
    if not db_path.exists():
        logger.error(f"Database not found: {db_path}")
        return
    
    conn = sqlite3.connect(db_path)
    
    # 1. 获取数据库中的股票代码
    codes_df = pd.read_sql("SELECT DISTINCT code FROM daily_kline", conn)
    db_codes = set(codes_df['code'].astype(str).str.strip().tolist())
    logger.info(f"Stocks in DB: {len(db_codes)}")
    
    # 2. 分页获取东方财富数据
    all_records = []
    page = 1
    max_pages = 60  # 约 6000 stocks / 100 per page
    
    logger.info("Fetching data from EastMoney API via curl...")
    while page <= max_pages:
        for attempt in range(3):
            records = fetch_eastmoney_spot_curl(page=page, page_size=100)
            if records is not None:
                break
            logger.warning(f"Page {page} retry {attempt+1}/3...")
            time.sleep(1)
        
        if records is None:
            logger.warning(f"Page {page} failed after retries, skipping to next page.")
            page += 1
            continue
        
        all_records.extend(records)
        logger.info(f"Page {page}: fetched {len(records)} stocks (total: {len(all_records)})")
        
        if len(records) < 100:
            break  # 最后一页
        
        page += 1
        time.sleep(5)  # 每页间隔5秒，避免触发东方财富反爬虫限流
    
    logger.success(f"Total fetched: {len(all_records)} stocks")
    
    # 3. 只保留数据库中已有的股票
    supplement = [r for r in all_records if r['code'] in db_codes]
    logger.info(f"Matched with DB: {len(supplement)} stocks")
    
    # 4. 更新数据库
    cursor = conn.cursor()
    updated = 0
    
    for r in supplement:
        cursor.execute(
            "UPDATE daily_kline SET turnover = ?, circulating_market_cap = ? WHERE code = ?",
            (r['turnover'], r['market_cap'], r['code'])
        )
        updated += cursor.rowcount
    
    conn.commit()
    conn.close()
    
    logger.success(f"Updated {updated} records")
    
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
    
    logger.info(f"Database stats after update:")
    logger.info(f"  Total records: {stats['total'].iloc[0]}")
    logger.info(f"  With turnover: {stats['has_turnover'].iloc[0]} ({stats['has_turnover'].iloc[0]/stats['total'].iloc[0]*100:.1f}%)")
    logger.info(f"  With market_cap: {stats['has_mc'].iloc[0]} ({stats['has_mc'].iloc[0]/stats['total'].iloc[0]*100:.1f}%)")
    
    return len(supplement)


if __name__ == "__main__":
    db_file = Path(__file__).parent / "data_pipeline" / "ashare_data.db"
    supplement_from_eastmoney(db_file)
