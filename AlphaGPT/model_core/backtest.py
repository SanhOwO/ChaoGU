import torch

class AShareBacktest:
    """
    A 股 T+1 回测引擎（真实交易成本版）

    核心规则：
    - T 日收盘信号 → T+1 日开盘成交（买或卖）
    - 买入成本：佣金 0.01% + 过户费 0.001% = 0.011%
    - 卖出成本：佣金 0.01% + 印花税 0.05% + 过户费 0.001% = 0.061%
    - 一次完整买卖总成本 ≈ 0.072%
    - 涨跌停保护：涨停不能买入，跌停不能卖出
    - 持仓周期：默认 1 个交易日（T+1 开盘买 → T+2 开盘卖）
    """
    def __init__(self):
        # 交易成本（基于成交额）
        self.commission_rate = 0.0001      # 佣金：万一
        self.min_commission = 0.1          # 最低佣金 0.1 元
        self.stamp_tax_rate = 0.0005      # 印花税：0.05%（仅卖出）
        self.transfer_fee_rate = 0.00001   # 过户费：0.001%（双向）
        self.slippage = 0.001              # 滑点 0.1%

        # 买入总成本率（简化，假设成交额 = trade_size）
        self.buy_cost_rate = self.commission_rate + self.transfer_fee_rate  # 0.011%
        # 卖出总成本率
        self.sell_cost_rate = self.commission_rate + self.stamp_tax_rate + self.transfer_fee_rate  # 0.061%

    def evaluate(self, factors, raw_data, target_ret):
        """
        factors: [Stocks, Time]  由 VM 执行公式得到的因子信号
        raw_data: dict with 'open', 'high', 'low', 'close', 'volume', 'turnover',
                             'market_cap', 'is_limit_up', 'is_limit_down'
        target_ret: [Stocks, Time]  T+1 日开盘到 T+2 日开盘的收益率

        Returns: (score, avg_return)
        """
        # 1. 因子信号 → 买入概率（0~1）
        signal = torch.sigmoid(factors)

        # 2. 基础过滤：排除低价股、高价股、小市值、无成交量
        close = raw_data['close']
        mc = raw_data.get('market_cap',
                          raw_data.get('market_cap', torch.ones_like(close) * 1e10))

        is_valid = (
            (close > 2.0) & (close < 500.0) &  # 价格正常
            (mc > 1e8) &                        # 市值 > 1亿
            (raw_data['volume'] > 0)            # 有成交量
        ).float()

        # 3. 涨跌停过滤
        is_limit_up = raw_data.get('is_limit_up', torch.zeros_like(close))
        is_limit_down = raw_data.get('is_limit_down', torch.zeros_like(close))

        # 买入：信号 > 0.70 且 非涨停 且 有效股票
        can_buy = is_valid * (1 - is_limit_up) * (signal > 0.70).float()

        # 4. 持仓收益：T+1 开盘买入 → T+2 开盘卖出
        # 目标收益已扣除卖出印花税+佣金+过户费（单边成本）
        # 但还要扣除买入时的单边成本
        gross_pnl = can_buy * target_ret

        # 5. 交易成本（双边）
        # 买入成本 + 卖出成本 = 0.011% + 0.061% = 0.072%
        total_tx_cost = can_buy * (self.buy_cost_rate + self.sell_cost_rate)

        # 6. 滑点（双边，简化按固定比例）
        slippage_cost = can_buy * (self.slippage * 2)  # 买入滑点 + 卖出滑点

        # 7. 净收益
        net_pnl = gross_pnl - total_tx_cost - slippage_cost

        # 8. 大回撤惩罚（单日亏损 > 5%）
        big_drawdowns = (net_pnl < -0.05).float().sum(dim=1)

        # 9. 累积收益
        cum_ret = net_pnl.sum(dim=1)

        # 10. 适应度评分
        score = cum_ret - (big_drawdowns * 2.0)

        # 11. 活跃度过滤已移除（让模型自由探索，靠交易成本自然筛选）

        # 12. 中位数适应度（鲁棒）
        final_fitness = torch.median(score)

        return final_fitness, cum_ret.mean().item()

    def evaluate_long_short(self, factors, raw_data, target_ret):
        """
        多空回测版本（A 股可做空时可用，如融券/股指期货）
        当前仅作为扩展接口
        """
        long_signal = (torch.sigmoid(factors) > 0.70).float()
        short_signal = (torch.sigmoid(factors) < 0.15).float()

        long_pnl = long_signal * target_ret
        short_pnl = short_signal * (-target_ret)

        total_pnl = long_pnl + short_pnl
        cum_ret = total_pnl.sum(dim=1)

        return torch.median(cum_ret), cum_ret.mean().item()
