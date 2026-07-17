import torch

class AShareBacktest:
    """
    A 股动态持仓回测引擎（真实交易成本版）

    核心规则：
    - T 日收盘信号 → T+1 日开盘成交（买或卖）
    - 买入成本：佣金 0.01% + 过户费 0.001% = 0.011%
    - 卖出成本：佣金 0.01% + 印花税 0.05% + 过户费 0.001% = 0.061%
    - 一次完整买卖总成本 ≈ 0.072%
    - 涨跌停保护：涨停不能买入，跌停不能卖出
    - 持仓周期：动态（signal > buy_threshold 买入，signal < sell_threshold 卖出）
    """
    def __init__(self, buy_threshold=0.70, sell_threshold=0.30, max_hold_days=60):
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold
        self.max_hold_days = max_hold_days
        
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
        # 一次完整买卖总成本
        self.total_cost_per_trade = self.buy_cost_rate + self.sell_cost_rate + self.slippage * 2

    def evaluate(self, factors, raw_data, target_ret=None):
        """
        factors: [Stocks, Time]  由 VM 执行公式得到的因子信号
        raw_data: dict with 'open', 'high', 'low', 'close', 'volume', 'turnover',
                             'market_cap', 'is_limit_up', 'is_limit_down'
        target_ret: 兼容旧接口，动态模式下不使用

        Returns: (score, avg_return)
        """
        # 1. 因子信号 → 买入概率（0~1）
        signal = torch.sigmoid(factors)
        n_stocks, n_time = signal.shape
        
        # 2. 获取原始数据
        op = raw_data['open']
        vol = raw_data['volume']
        close = raw_data['close']
        mc = raw_data.get('market_cap', torch.ones_like(close) * 1e10)

        # 3. 基础过滤：排除低价股、高价股、小市值、无成交量
        is_valid = (
            (close > 2.0) & (close < 500.0) &  # 价格正常
            (mc > 1e8) &                        # 市值 > 1亿
            (vol > 0)                            # 有成交量
        ).float()

        # 4. 涨跌停过滤
        is_limit_up = raw_data.get('is_limit_up', torch.zeros_like(close))
        is_limit_down = raw_data.get('is_limit_down', torch.zeros_like(close))

        # 5. 逐股逐日状态化回测（动态持仓）
        all_pnl = []
        
        for s in range(n_stocks):
            in_position = False
            entry_price = 0.0
            days_held = 0
            stock_pnl = 0.0
            
            for t in range(n_time - 1):
                can_trade = is_valid[s, t].item() > 0.5
                
                if not in_position:
                    # 尝试买入：signal > buy_threshold 且 非涨停 且 有效股票
                    if (can_trade and 
                        is_limit_up[s, t].item() < 0.5 and 
                        signal[s, t] > self.buy_threshold):
                        entry_price = op[s, t + 1].item()
                        in_position = True
                        days_held = 0
                else:
                    days_held += 1
                    
                    # 判断卖出条件
                    should_exit = False
                    if signal[s, t] < self.sell_threshold:
                        should_exit = True
                    elif days_held >= self.max_hold_days:
                        should_exit = True
                    
                    # 执行卖出：非跌停
                    if should_exit and is_limit_down[s, t].item() < 0.5:
                        exit_price = op[s, t + 1].item()
                        trade_ret = (exit_price / (entry_price + 1e-9)) - 1.0
                        net_ret = trade_ret - self.total_cost_per_trade
                        stock_pnl += net_ret
                        in_position = False
            
            # 如果到数据末尾仍持仓，强制平仓
            if in_position:
                exit_price = op[s, -1].item()
                trade_ret = (exit_price / (entry_price + 1e-9)) - 1.0
                net_ret = trade_ret - self.total_cost_per_trade
                stock_pnl += net_ret
            
            all_pnl.append(stock_pnl)
        
        # 6. 适应度评分：中位数累积收益（鲁棒）
        pnl_tensor = torch.tensor(all_pnl, dtype=torch.float32, device=factors.device)
        score = torch.median(pnl_tensor)
        avg_return = pnl_tensor.mean().item()
        
        return score, avg_return
    
    def evaluate_with_trades(self, factors, raw_data):
        """
        返回详细的交易日志，用于分析和可视化
        
        Returns: (score, avg_return, trades_list)
            trades_list: list of dict with keys:
                entry_idx, exit_idx, days_held, entry_price, exit_price, raw_return, net_return, exit_reason
        """
        signal = torch.sigmoid(factors)
        n_stocks, n_time = signal.shape
        
        op = raw_data['open']
        vol = raw_data['volume']
        close = raw_data['close']
        mc = raw_data.get('market_cap', torch.ones_like(close) * 1e10)
        
        is_valid = (
            (close > 2.0) & (close < 500.0) &
            (mc > 1e8) &
            (vol > 0)
        ).float()
        
        is_limit_up = raw_data.get('is_limit_up', torch.zeros_like(close))
        is_limit_down = raw_data.get('is_limit_down', torch.zeros_like(close))
        
        all_trades = []
        all_pnl = []
        
        for s in range(n_stocks):
            in_position = False
            entry_price = 0.0
            entry_date_idx = 0
            days_held = 0
            stock_pnl = 0.0
            
            for t in range(n_time - 1):
                can_trade = is_valid[s, t].item() > 0.5
                
                if not in_position:
                    if (can_trade and 
                        is_limit_up[s, t].item() < 0.5 and 
                        signal[s, t] > self.buy_threshold):
                        entry_price = op[s, t + 1].item()
                        entry_date_idx = t
                        in_position = True
                        days_held = 0
                else:
                    days_held += 1
                    
                    exit_reason = None
                    if signal[s, t] < self.sell_threshold:
                        should_exit = True
                        exit_reason = 'signal'
                    elif days_held >= self.max_hold_days:
                        should_exit = True
                        exit_reason = 'max_hold'
                    else:
                        should_exit = False
                    
                    if should_exit and is_limit_down[s, t].item() < 0.5:
                        exit_price = op[s, t + 1].item()
                        trade_ret = (exit_price / (entry_price + 1e-9)) - 1.0
                        net_ret = trade_ret - self.total_cost_per_trade
                        stock_pnl += net_ret
                        in_position = False
                        
                        all_trades.append({
                            'stock_idx': s,
                            'entry_idx': entry_date_idx,
                            'exit_idx': t,
                            'days_held': days_held,
                            'entry_price': entry_price,
                            'exit_price': exit_price,
                            'raw_return': trade_ret,
                            'net_return': net_ret,
                            'exit_reason': exit_reason,
                        })
            
            if in_position:
                exit_price = op[s, -1].item()
                trade_ret = (exit_price / (entry_price + 1e-9)) - 1.0
                net_ret = trade_ret - self.total_cost_per_trade
                stock_pnl += net_ret
                in_position = False
                
                all_trades.append({
                    'stock_idx': s,
                    'entry_idx': entry_date_idx,
                    'exit_idx': n_time - 1,
                    'days_held': n_time - 1 - entry_date_idx,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'raw_return': trade_ret,
                    'net_return': net_ret,
                    'exit_reason': 'end_of_data',
                })
            
            all_pnl.append(stock_pnl)
        
        pnl_tensor = torch.tensor(all_pnl, dtype=torch.float32, device=factors.device)
        score = torch.median(pnl_tensor)
        avg_return = pnl_tensor.mean().item()
        
        return score, avg_return, all_trades

    def evaluate_long_short(self, factors, raw_data, target_ret):
        """
        多空回测版本（A 股可做空时可用，如融券/股指期货）
        当前仅作为扩展接口，仍使用固定1天逻辑
        """
        long_signal = (torch.sigmoid(factors) > 0.70).float()
        short_signal = (torch.sigmoid(factors) < 0.15).float()

        long_pnl = long_signal * target_ret
        short_pnl = short_signal * (-target_ret)

        total_pnl = long_pnl + short_pnl
        cum_ret = total_pnl.sum(dim=1)

        return torch.median(cum_ret), cum_ret.mean().item()
