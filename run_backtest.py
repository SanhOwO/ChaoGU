#!/usr/bin/env python3
"""
回测最优公式 [0, 10, 10, 10, 20, 19, 22, 6, 9, 19, 21, 17, 15]
输出详细的交易记录和收益曲线
"""
import sys
import torch
import pandas as pd
from datetime import datetime

# 确保能导入 model_core
sys.path.insert(0, r'G:\学习理财赚钱\量化\AI\AlphaGPT个股策略')
from AlphaGPT.model_core.data_loader import AShareDataLoader
from AlphaGPT.model_core.vm import StackVM
from AlphaGPT.model_core.backtest import AShareBacktest
from AlphaGPT.model_core.config import ModelConfig

# ============ 最优公式 ============
BEST_FORMULA = [0, 10, 10, 10, 20, 19, 22, 6, 9, 19, 21, 17, 15]

# ============ 回测配置 ============
START_DATE = '2023-01-01'
STOCK_CODE = '002549'

print("=" * 60)
print(f"最优公式回测: {STOCK_CODE}")
print(f"训练区间: {START_DATE} ~ 最新")
print(f"公式: {BEST_FORMULA}")
print("=" * 60)

# 1. 加载数据
loader = AShareDataLoader()
loader.load_data(start_date=START_DATE, specific_codes=[STOCK_CODE])

# 2. 执行公式
vm = StackVM()
factors = vm.execute(BEST_FORMULA, loader.feat_tensor)

if factors is None:
    print("公式执行失败！")
    sys.exit(1)

print(f"\n公式信号 shape: {factors.shape}")
print(f"信号统计: min={factors.min():.4f}, max={factors.max():.4f}, mean={factors.mean():.4f}, std={factors.std():.4f}")

# 3. 用 backtest 评估
bt = AShareBacktest()
score, avg_ret = bt.evaluate(factors, loader.raw_data_cache, loader.target_ret)

print(f"\n{'='*60}")
print(f"回测评分: {score:.4f}")
print(f"平均收益率: {avg_ret:.4f} ({avg_ret*100:.2f}%)")
print(f"{'='*60}")

# 4. 详细的逐日交易记录
print("\n" + "=" * 80)
print("逐日交易记录")
print("=" * 80)

signal = torch.sigmoid(factors)
can_buy = (signal > 0.70).float()

# 逐天分析
stock_idx = 0  # 只有一只股票
dates = loader.dates
close = loader.raw_data_cache['close'][stock_idx].cpu().numpy()
open_p = loader.raw_data_cache['open'][stock_idx].cpu().numpy()
high = loader.raw_data_cache['high'][stock_idx].cpu().numpy()
low = loader.raw_data_cache['low'][stock_idx].cpu().numpy()
volume = loader.raw_data_cache['volume'][stock_idx].cpu().numpy()
target_ret = loader.target_ret[stock_idx].cpu().numpy()
sig = signal[stock_idx].cpu().numpy()
buy_mask = can_buy[stock_idx].cpu().numpy()

# 计算逐笔收益
pnl_list = []
cum_pnl = 0.0
trades = []

for t in range(len(dates) - 2):  # 最后两天无法交易
    date = dates[t]
    sig_val = sig[t]
    is_buy = buy_mask[t] > 0
    ret = target_ret[t]
    
    if is_buy:
        # 交易成本
        buy_cost = 0.00011  # 佣金+过户费
        sell_cost = 0.00061  # 佣金+印花税+过户费
        slippage = 0.001 * 2  # 双边滑点
        total_cost = buy_cost + sell_cost + slippage
        
        net_ret = ret - total_cost
        cum_pnl += net_ret
        
        trades.append({
            'date': date,
            'close': close[t],
            'signal': sig_val,
            'ret_next': ret,
            'net_ret': net_ret,
            'cum_pnl': cum_pnl,
            'notes': 'BUY'
        })
        pnl_list.append(net_ret)
    else:
        trades.append({
            'date': date,
            'close': close[t],
            'signal': sig_val,
            'ret_next': ret,
            'net_ret': 0.0,
            'cum_pnl': cum_pnl,
            'notes': '-'
        })

# 打印交易记录（只打印买入的日子，节省空间）
print(f"\n{'日期':<12} {'收盘价':>8} {'信号':>8} {'次日收益':>10} {'净收益':>10} {'累计':>10} {'操作':>6}")
print("-" * 80)

buy_count = 0
win_count = 0
loss_count = 0
for tr in trades:
    if tr['notes'] == 'BUY':
        buy_count += 1
        if tr['net_ret'] > 0:
            win_count += 1
        else:
            loss_count += 1
        print(f"{tr['date']:<12} {tr['close']:>8.2f} {tr['signal']:>8.4f} {tr['ret_next']*100:>9.2f}% {tr['net_ret']*100:>9.2f}% {tr['cum_pnl']*100:>9.2f}% {tr['notes']:>6}")

print("-" * 80)
print(f"\n总交易天数: {len(dates)}")
print(f"买入次数: {buy_count}")
print(f"胜率: {win_count}/{buy_count} = {win_count/buy_count*100:.1f}%" if buy_count > 0 else "胜率: N/A")
print(f"总净收益: {cum_pnl*100:.2f}%")
print(f"平均每次收益: {cum_pnl/buy_count*100:.2f}%" if buy_count > 0 else "平均每次收益: N/A")

# 5. 保存到 CSV
df = pd.DataFrame(trades)
output_path = r'G:\学习理财赚钱\量化\AI\AlphaGPT个股策略\backtest_002549_best_formula.csv'
df.to_csv(output_path, index=False, encoding='utf-8-sig')
print(f"\n详细交易记录已保存到: {output_path}")
