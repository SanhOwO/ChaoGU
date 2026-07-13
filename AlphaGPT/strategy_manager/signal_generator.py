import json
import torch
import pandas as pd
from datetime import datetime
from loguru import logger
from pathlib import Path

from model_core.vm import StackVM
from model_core.data_loader import AShareDataLoader
from strategy_manager.config import AShareStrategyConfig

class AShareSignalGenerator:
    """
    A 股每日收盘信号生成器
    
    T 日 15:00 收盘后运行：
    1. 加载最新数据
    2. 使用最优公式对所有股票打分
    3. 生成人工操作参考报告
    """
    
    def __init__(self, formula_path="best_meme_strategy.json"):
        self.config = AShareStrategyConfig()
        self.vm = StackVM()
        self.loader = AShareDataLoader()
        self.formula_path = formula_path
        self.formula = None
        self._load_formula()
    
    def _load_formula(self):
        """加载最优公式"""
        try:
            with open(self.formula_path, "r") as f:
                data = json.load(f)
                self.formula = data if isinstance(data, list) else data.get("formula")
            logger.success(f"Loaded formula: {self.formula}")
        except FileNotFoundError:
            logger.critical(f"Formula file not found: {self.formula_path}")
            logger.info("Please run training first: python -m model_core.engine")
            raise
    
    def generate_signals(self, top_n=20):
        """
        生成全市场信号
        
        Returns:
            DataFrame: 包含信号分数、因子值的报告
        """
        # 1. 加载最新数据快照
        logger.info("Loading latest market data...")
        self.loader.load_data(limit_stocks=None, min_history=60)
        feat_slice, codes, latest_date = self.loader.get_latest_snapshot(n_days=60)
        
        # 2. 构建 token 映射
        token_map = {code: idx for idx, code in enumerate(codes)}
        
        # 3. 执行公式：获取全市场最新信号
        logger.info("Running formula inference...")
        raw_signals = self.vm.execute(self.formula, feat_slice)
        
        if raw_signals is None:
            logger.error("Formula execution failed!")
            return pd.DataFrame()
        
        # 4. 取最新一天的信号
        latest_signals = raw_signals[:, -1]
        scores = torch.sigmoid(latest_signals).cpu().numpy()
        
        # 5. 获取每只股票的最新信息
        records = []
        for idx, code in enumerate(codes):
            info = self.loader.get_stock_info(code)
            if info is None:
                continue
            
            # 计算关键因子值（用于人工参考）
            feat_values = feat_slice[idx, :, -1].cpu().numpy()
            
            records.append({
                'code': code,
                'score': float(scores[idx]),
                'close': info['latest_close'],
                'volume': info['latest_volume'],
                'market_cap': info['market_cap'] / 1e8,  # 亿元
                'ret': float(feat_values[0]),
                'turn': float(feat_values[1]),
                'volat': float(feat_values[2]),
                'mom': float(feat_values[3]),
                'rsi': float(feat_values[5]) * 50 + 50,  # 还原为 0-100
                'dev': float(feat_values[6]),
                'vr': float(feat_values[7]),
                'macd_hist': float(feat_values[11]),
            })
        
        df = pd.DataFrame(records)
        
        # 6. 过滤
        # 排除成交额过低（流动性不足）
        df = df[df['volume'] * df['close'] >= self.config.MIN_DAILY_AMOUNT]
        
        # 按信号分数排序
        df = df.sort_values('score', ascending=False).reset_index(drop=True)
        
        # 7. 标记操作建议
        df['suggestion'] = df['score'].apply(self._suggestion)
        
        return df, latest_date
    
    def _suggestion(self, score):
        """根据分数给出操作建议"""
        if score >= self.config.BUY_THRESHOLD + 0.05:
            return "🔥 强烈关注"
        elif score >= self.config.BUY_THRESHOLD:
            return "✅ 建议关注"
        elif score >= 0.6:
            return "🟡 可观察"
        elif score <= self.config.SELL_THRESHOLD:
            return "⚠️ 建议回避"
        else:
            return "➖ 中性"
    
    def generate_daily_report(self, output_path=None):
        """
        生成 T+1 日操作参考报告
        
        Returns:
            report_path: 报告文件路径
        """
        df, latest_date = self.generate_signals()
        
        if df.empty:
            logger.error("No signals generated.")
            return None
        
        # 生成报告文本
        lines = []
        lines.append("=" * 60)
        lines.append(f"  AlphaGPT-AShare 每日信号报告")
        lines.append(f"  分析日期: {latest_date} (T 日)")
        lines.append(f"  操作日期: T+1 日开盘")
        lines.append("=" * 60)
        lines.append("")
        
        # Top 买入信号
        buy_signals = df[df['score'] >= self.config.BUY_THRESHOLD].head(self.config.TOP_N_STOCKS)
        lines.append(f"📈 【重点关注】Top 买入信号（分数 >= {self.config.BUY_THRESHOLD}）")
        lines.append("-" * 60)
        
        if not buy_signals.empty:
            for i, row in buy_signals.iterrows():
                lines.append(f"\n  Rank {i+1}: {row['code']}")
                lines.append(f"    信号分数: {row['score']:.4f}  {row['suggestion']}")
                lines.append(f"    最新收盘价: {row['close']:.2f} 元")
                lines.append(f"    流通市值: {row['market_cap']:.2f} 亿")
                lines.append(f"    20日动量: {row['mom']:.3f}  14日RSI: {row['rsi']:.1f}")
                lines.append(f"    偏离均线: {row['dev']:.3f}  量比: {row['vr']:.2f}")
                lines.append(f"    MACD柱: {row['macd_hist']:.3f}  波动率: {row['volat']:.3f}")
        else:
            lines.append("  今日无强烈买入信号，建议观望。")
        
        lines.append("")
        lines.append("-" * 60)
        lines.append("")
        
        # 回避信号
        sell_signals = df[df['score'] <= self.config.SELL_THRESHOLD].head(10)
        lines.append(f"📉 【回避信号】分数 <= {self.config.SELL_THRESHOLD}")
        if not sell_signals.empty:
            for i, row in sell_signals.iterrows():
                lines.append(f"  {row['code']}: 分数 {row['score']:.4f}  {row['suggestion']}")
        else:
            lines.append("  今日无强烈回避信号。")
        
        lines.append("")
        lines.append("-" * 60)
        lines.append("")
        
        # 统计信息
        lines.append("📊 【市场统计】")
        lines.append(f"  全市场分析股票数: {len(df)}")
        lines.append(f"  平均信号分数: {df['score'].mean():.4f}")
        lines.append(f"  中位信号分数: {df['score'].median():.4f}")
        lines.append(f"  分数标准差: {df['score'].std():.4f}")
        lines.append(f"  最高分: {df['score'].max():.4f} ({df.iloc[0]['code']})")
        lines.append(f"  最低分: {df['score'].min():.4f} ({df.iloc[-1]['code']})")
        
        lines.append("")
        lines.append("=" * 60)
        lines.append("⚠️ 免责声明：本报告由 AI 模型生成，仅供研究参考，不构成投资建议。")
        lines.append("   操作风险自负，请结合自身判断决策。")
        lines.append("=" * 60)
        
        report_text = "\n".join(lines)
        
        # 保存报告
        if output_path is None:
            output_path = f"daily_report_{latest_date}.txt"
        
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(report_text)
        
        # 同时保存 CSV 明细
        csv_path = output_path.replace(".txt", "_detail.csv")
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        
        logger.success(f"Report saved: {output_path}")
        logger.success(f"Detail CSV saved: {csv_path}")
        
        return output_path, df


if __name__ == "__main__":
    # 测试运行
    gen = AShareSignalGenerator()
    report_path, df = gen.generate_daily_report()
    print(f"\nReport: {report_path}")
    print(df.head(10)[['code', 'score', 'suggestion', 'close']].to_string(index=False))
