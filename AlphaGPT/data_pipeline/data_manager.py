import sqlite3
import pandas as pd
from datetime import datetime, timedelta
from loguru import logger
from .config import AShareConfig
from .providers.akshare_provider import AKShareProvider

class AShareDataManager:
    """A 股数据管理器：AKShare → SQLite"""
    
    def __init__(self):
        self.config = AShareConfig()
        self.provider = AKShareProvider()
        self.db_path = self.config.DB_PATH
        self._init_db()
    
    def _init_db(self):
        """初始化 SQLite 数据库"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 日 K 数据表
        cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS {self.config.TABLE_DAILY} (
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
        
        # 创建索引加速查询
        cursor.execute(f"""
            CREATE INDEX IF NOT EXISTS idx_code ON {self.config.TABLE_DAILY} (code)
        """)
        cursor.execute(f"""
            CREATE INDEX IF NOT EXISTS idx_date ON {self.config.TABLE_DAILY} (date)
        """)
        
        # 元数据表：记录最后更新日期
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS meta (
                key TEXT PRIMARY KEY,
                value TEXT
            )
        """)
        
        conn.commit()
        conn.close()
        logger.info(f"Database initialized: {self.db_path}")
    
    def get_last_update_date(self):
        """获取最后更新日期"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT value FROM meta WHERE key = 'last_update'")
        row = cursor.fetchone()
        conn.close()
        return row[0] if row else None
    
    def set_last_update_date(self, date_str):
        """设置最后更新日期"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES ('last_update', ?)",
            (date_str,)
        )
        conn.commit()
        conn.close()
    
    def pipeline_sync_daily(self):
        """
        每日数据同步主流程
        """
        logger.info("=" * 50)
        logger.info("A-Share Daily Data Sync Started")
        logger.info("=" * 50)
        
        # 1. 获取全市场股票列表
        logger.info("Step 1: Fetching stock list from AKShare...")
        stock_list = self.provider.get_all_stock_list()
        
        if stock_list.empty:
            logger.error("Failed to fetch stock list!")
            return
        
        logger.info(f"Total stocks: {len(stock_list)}")
        
        # 2. 过滤
        filtered = stock_list[
            (stock_list['price'] >= self.config.MIN_PRICE) &
            (stock_list['price'] <= self.config.MAX_PRICE) &
            (stock_list['market_cap'] >= self.config.MIN_MARKET_CAP)
        ]
        
        # 排除 ST（名称中包含 ST）
        if self.config.EXCLUDE_ST:
            filtered = filtered[~filtered['name'].str.contains('ST', case=False, na=False)]
        
        codes = filtered['code'].tolist()
        logger.info(f"Stocks after filtering: {len(codes)}")
        
        # 3. 确定更新范围
        last_update = self.get_last_update_date()
        if last_update:
            # 增量更新：从上次更新日期+1天开始
            start_date = (datetime.strptime(last_update, '%Y-%m-%d') + timedelta(days=1)).strftime('%Y%m%d')
            logger.info(f"Incremental update from {start_date}")
        else:
            # 首次全量：下载 3 年历史
            start_date = (datetime.now() - timedelta(days=365 * self.config.HISTORY_YEARS)).strftime('%Y%m%d')
            logger.info(f"Full history update from {start_date}")
        
        end_date = datetime.now().strftime('%Y%m%d')
        
        # 4. 逐只获取历史数据
        all_records = []
        success_count = 0
        
        for i, code in enumerate(codes):
            if i % 100 == 0:
                logger.info(f"Progress: {i}/{len(codes)} stocks processed")
            
            df = self.provider.get_stock_history(code, start_date=start_date, end_date=end_date)
            
            if df.empty:
                continue
            
            # 填充流通市值（用 stock_list 中的最新市值）
            mc = filtered[filtered['code'] == code]['market_cap'].values
            if len(mc) > 0:
                df['circulating_market_cap'] = mc[0]
            
            all_records.append(df)
            success_count += 1
        
        logger.info(f"Data fetch complete. Success: {success_count}/{len(codes)}")
        
        # 5. 写入数据库
        if all_records:
            combined = pd.concat(all_records, ignore_index=True)
            
            # 去重
            combined = combined.drop_duplicates(subset=['date', 'code'], keep='last')
            
            conn = sqlite3.connect(self.db_path)
            
            # 使用 REPLACE 策略：如果存在则更新
            combined.to_sql(
                self.config.TABLE_DAILY,
                conn,
                if_exists='append',
                index=False,
                method='multi'
            )
            
            # 清理可能的重复（由于 append 模式）
            cursor = conn.cursor()
            cursor.execute(f"""
                DELETE FROM {self.config.TABLE_DAILY}
                WHERE rowid NOT IN (
                    SELECT MIN(rowid)
                    FROM {self.config.TABLE_DAILY}
                    GROUP BY date, code
                )
            """)
            
            conn.commit()
            conn.close()
            
            logger.info(f"Inserted/Updated {len(combined)} records")
        
        # 6. 更新元数据
        self.set_last_update_date(end_date[:4] + '-' + end_date[4:6] + '-' + end_date[6:])
        logger.success("Daily sync complete!")


if __name__ == "__main__":
    import sys
    manager = AShareDataManager()
    manager.pipeline_sync_daily()
