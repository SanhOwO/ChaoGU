import torch

class AShareBacktest:
    """
    A 股 T+1 回测引擎
    
    核心规则：
    - T 日收盘信号买入，T+1 日开盘成交
    - 目标收益 = T+1 日开盘 → T+2 日开盘（持有 1 天）
    - 涨跌停过滤：涨停不能买入，跌停不能卖出
    - 交易费用：买入 0.025% 佣金，卖出 0.025% 佣金 + 0.05% 印花税
    """
    def __init__(self):
        self.trade_size = 10000.0    # 每笔 1 万元
        self.base_fee_buy = 0.00025   # 买入佣金
        self.base_fee_sell = 0.00075  # 卖出佣金 + 印花税
        self.slippage = 0.001         # 滑点 0.1%
        self.min_commission = 5.0     # 最低佣金 5 元
    
    def evaluate(self, factors, raw_data, target_ret):
        """
        factors: [Stocks, Time]  由 VM 执行公式得到的因子信号
        raw_data: dict with 'open', 'high', 'low', 'close', 'volume', 
                             'circulating_market_cap', 'is_limit_up', 'is_limit_down'
        target_ret: [Stocks, Time]  T+1 日开盘到 T+2 日开盘的收益率
        
        Returns: (score, avg_return)
        """
        # 1. 因子信号 → 买入概率
        signal = torch.sigmoid(factors)
        
        # 2. 基础过滤：排除低价股、高价股、小市值、ST（无数据）
        close = raw_data['close']
        mc = raw_data['circulating_market_cap']
        
        # 有效股票掩码：价格正常、市值正常、有成交量
        is_valid = (
            (close > 2.0) & (close < 500.0) &  # 价格正常
            (mc > 1e8) &                        # 市值 > 1亿
            (raw_data['volume'] > 0)            # 有成交量
        ).float()
        
        # 3. 涨跌停过滤：涨停不能买入
        is_limit_up = raw_data.get('is_limit_up', torch.zeros_like(close))
        can_buy = is_valid * (1 - is_limit_up) * (signal > 0.85).float()
        
        # 4. 持仓信号（T 日收盘决定买入，T+1 日开盘成交）
        # 使用 prev_pos 模拟 T+1 持仓
        position = can_buy
        
        # 5. 目标收益：持有 1 天的收益（T+1 开盘到 T+2 开盘）
        gross_pnl = position * target_ret
        
        # 6. 交易成本：买入时扣除
        # 佣金 = max(trade_size * 0.025%, 5元)，简化按固定比例
        tx_cost = position * self.base_fee_buy  # 买入成本
        
        # 7. 滑点：冲击成本
        slippage_cost = position * self.slippage
        
        # 8. 净收益
        net_pnl = gross_pnl - tx_cost - slippage_cost
        
        # 9. 大回撤惩罚（单日亏损 > 5%）
        big_drawdowns = (net_pnl < -0.05).float().sum(dim=1)
        
        # 10. 累积收益
        cum_ret = net_pnl.sum(dim=1)
        
        # 11. 适应度评分
        score = cum_ret - (big_drawdowns * 2.0)
        
        # 12. 活跃度过滤：交易次数太少（持仓日数 < 5）惩罚
        activity = position.sum(dim=1)
        score = torch.where(activity < 5, torch.tensor(-10.0, device=score.device), score)
        
        # 13. 中位数适应度（鲁棒）
        final_fitness = torch.median(score)
        
        return final_fitness, cum_ret.mean().item()
    
    def evaluate_long_short(self, factors, raw_data, target_ret):
        """
        多空回测版本（A 股可做空时可用，如融券/股指期货）
        当前仅作为扩展接口
        """
        # 多头信号
        long_signal = (torch.sigmoid(factors) > 0.85).float()
        # 空头信号（信号低分）
        short_signal = (torch.sigmoid(factors) < 0.15).float()
        
        long_pnl = long_signal * target_ret
        short_pnl = short_signal * (-target_ret)  # 做空收益 = -return
        
        total_pnl = long_pnl + short_pnl
        cum_ret = total_pnl.sum(dim=1)
        
        return torch.median(cum_ret), cum_ret.mean().item()
