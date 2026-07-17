#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
个股回测脚本
用法: python run_backtest.py <股票代码>
例: python run_backtest.py 600150
"""

import sys
import os
import json
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'AlphaGPT'))

from model_core.data_loader import AShareDataLoader
from model_core.config import ModelConfig
from model_core.vm import StackVM
from model_core.backtest import AShareBacktest
from model_core.vocab import FORMULA_VOCAB
import torch


def decode_formula(formula_tokens):
    """将 token 序列翻译为可读名称"""
    token_names = FORMULA_VOCAB.token_names
    readable = []
    for t in formula_tokens:
        if t < len(token_names):
            readable.append(token_names[t])
        else:
            readable.append(f"UNK({t})")
    return readable


def run_backtest(stock_code):
    print(f">>> 回测股票: {stock_code}")
    print("=" * 70)
    
    # 1. 加载策略
    strategy_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "个股训练数据",
        f"{stock_code}_strategy.json"
    )
    if not os.path.exists(strategy_path):
        print(f"[ERROR] 策略文件不存在: {strategy_path}")
        return
    
    with open(strategy_path, "r") as f:
        formula_tokens = json.load(f)
    
    print(f"[OK] 策略公式 Token: {formula_tokens}")
    print(f"     公式解读: {' | '.join(decode_formula(formula_tokens))}")
    
    # 2. 加载数据
    print(f"\n[...] 加载数据...")
    loader = AShareDataLoader()
    loader.load_data(
        start_date=ModelConfig.TRAIN_START_DATE,
        end_date=ModelConfig.TRAIN_END_DATE,
        specific_codes=[stock_code]
    )
    dates = loader.dates
    print(f"[OK] 数据加载: {len(dates)} 个交易日 ({dates[0]} ~ {dates[-1]})")
    
    # 3. 执行公式
    vm = StackVM()
    res = None
    used_len = None
    for trunc_len in [5, 7, 9, 11, 13]:
        if trunc_len > len(formula_tokens):
            continue
        candidate = formula_tokens[:trunc_len]
        res = vm.execute(candidate, loader.feat_tensor)
        if res is not None:
            used_len = trunc_len
            break
    
    if res is None:
        print("[ERROR] 公式执行失败")
        return
    
    print(f"[OK] 公式执行成功 (前缀长度: {used_len})")
    
    # 4. 回测（动态持仓）
    bt = AShareBacktest(buy_threshold=0.70, sell_threshold=0.30, max_hold_days=60)
    score, avg_ret, trades = bt.evaluate_with_trades(res, loader.raw_data_cache)
    
    print(f"\n{'=' * 70}")
    print(f"📊 回测结果汇总")
    print(f"{'=' * 70}")
    print(f"  评分 (中位数累积收益): {score:.4f}")
    print(f"  平均累积收益: {avg_ret:.4f}")
    print(f"  总交易次数: {len(trades)}")
    
    if not trades:
        print("  [WARN] 无交易记录")
        return
    
    # 5. 交易统计
    net_rets = [t['net_return'] for t in trades]
    raw_rets = [t['raw_return'] for t in trades]
    hold_days = [t['days_held'] for t in trades]
    
    wins = [r for r in net_rets if r > 0]
    losses = [r for r in net_rets if r <= 0]
    
    print(f"\n【收益统计】")
    print(f"  总净收益: {sum(net_rets):+.4f} ({sum(net_rets):+.2%})")
    print(f"  单笔平均净收益: {np.mean(net_rets):+.4f}")
    print(f"  单笔最大盈利: {max(net_rets):+.4f} ({max(net_rets):+.2%})")
    print(f"  单笔最大亏损: {min(net_rets):+.4f} ({min(net_rets):+.2%})")
    
    print(f"\n【胜率与盈亏比】")
    print(f"  盈利次数: {len(wins)}")
    print(f"  亏损次数: {len(losses)}")
    print(f"  胜率: {len(wins)/len(trades)*100:.1f}%")
    if losses:
        avg_win = np.mean(wins) if wins else 0
        avg_loss = abs(np.mean(losses))
        print(f"  平均盈利: {avg_win:+.4f}")
        print(f"  平均亏损: {avg_loss:+.4f}")
        print(f"  盈亏比: {avg_win/avg_loss:.2f}")
    
    print(f"\n【持仓周期】")
    print(f"  平均持仓天数: {np.mean(hold_days):.1f}")
    print(f"  最短持仓: {min(hold_days)} 天")
    print(f"  最长持仓: {max(hold_days)} 天")
    
    # 6. 时间序列分析：累计收益曲线（增加退出原因+价格列）
    print(f"\n【累计收益走势 (最近20笔交易)】")
    cumsum = 0
    print(f"{'#':>3} {'日期':<12} {'持仓':>4} {'买入价':>10} {'卖出价':>10} {'原始收益':>10} {'净收益':>10} {'累计':>10} {'退出原因':>10}")
    print("-" * 92)
    for i, t in enumerate(trades[-20:], 1):
        entry_date = dates[t['entry_idx']] if t['entry_idx'] < len(dates) else "N/A"
        cumsum += t['net_return']
        reason_map = {
            'signal': '信号卖出',
            'max_hold': '【强平】',
            'end_of_data': '数据末尾'
        }
        reason = reason_map.get(t.get('exit_reason', 'unknown'), t.get('exit_reason', 'unknown'))
        print(f"{i:>3} {entry_date:<12} {t['days_held']:>4}d {t['entry_price']:>10.2f} {t['exit_price']:>10.2f} {t['raw_return']:>+9.2%} {t['net_return']:>+9.2%} {cumsum:>+9.2%} {reason:>10}")
    
    # 6.5 专门列出触发 max_hold_days 的交易
    max_hold_trades = [t for t in trades if t.get('exit_reason') == 'max_hold']
    if max_hold_trades:
        print(f"\n{'=' * 92}")
        print(f"⚠️  触发 max_hold_days ({bt.max_hold_days}天) 强制平仓的交易")
        print(f"{'=' * 92}")
        print(f"{'#':>3} {'买入日期':<12} {'卖出日期':<12} {'持仓':>4} {'买入价':>10} {'卖出价':>10} {'原始收益':>10} {'净收益':>10}")
        print("-" * 92)
        for i, t in enumerate(max_hold_trades, 1):
            entry_date = dates[t['entry_idx']] if t['entry_idx'] < len(dates) else "N/A"
            exit_date = dates[t['exit_idx']] if t['exit_idx'] < len(dates) else "N/A"
            print(f"{i:>3} {entry_date:<12} {exit_date:<12} {t['days_held']:>4}d {t['entry_price']:>10.2f} {t['exit_price']:>10.2f} {t['raw_return']:>+9.2%} {t['net_return']:>+9.2%}")
        print(f"\n  强制平仓次数: {len(max_hold_trades)} / {len(trades)} ({len(max_hold_trades)/len(trades)*100:.1f}%)")
        
        # 强制平仓的收益统计
        mh_rets = [t['net_return'] for t in max_hold_trades]
        mh_wins = [r for r in mh_rets if r > 0]
        print(f"  强平交易胜率: {len(mh_wins)/len(max_hold_trades)*100:.1f}%")
        print(f"  强平交易平均收益: {np.mean(mh_rets):+.4f} ({np.mean(mh_rets):+.2%})")
    else:
        print(f"\n✅ 没有交易触发 max_hold_days 强制平仓")
    
    # 退出原因统计
    print(f"\n【退出原因分布】")
    reason_counts = {}
    for t in trades:
        r = t.get('exit_reason', 'unknown')
        reason_counts[r] = reason_counts.get(r, 0) + 1
    reason_names = {'signal': '信号触发卖出', 'max_hold': '强制平仓(满60天)', 'end_of_data': '数据末尾强制平'}
    for r, cnt in reason_counts.items():
        print(f"  {reason_names.get(r, r)}: {cnt} 次 ({cnt/len(trades)*100:.1f}%)")
    print(f"\n【累计收益走势 (最近20笔交易)】")
    cumsum = 0
    print(f"{'#':>3} {'日期':<12} {'持仓':>4} {'原始收益':>10} {'净收益':>10} {'累计':>10} {'退出原因':>10}")
    print("-" * 72)
    for i, t in enumerate(trades[-20:], 1):
        entry_date = dates[t['entry_idx']] if t['entry_idx'] < len(dates) else "N/A"
        cumsum += t['net_return']
        reason_map = {
            'signal': '信号卖出',
            'max_hold': '【强平】',
            'end_of_data': '数据末尾'
        }
        reason = reason_map.get(t.get('exit_reason', 'unknown'), t.get('exit_reason', 'unknown'))
        print(f"{i:>3} {entry_date:<12} {t['days_held']:>4}d {t['raw_return']:>+9.2%} {t['net_return']:>+9.2%} {cumsum:>+9.2%} {reason:>10}")
    
    # 6.5 专门列出触发 max_hold_days 的交易
    max_hold_trades = [t for t in trades if t.get('exit_reason') == 'max_hold']
    if max_hold_trades:
        print(f"\n{'=' * 72}")
        print(f"⚠️  触发 max_hold_days ({bt.max_hold_days}天) 强制平仓的交易")
        print(f"{'=' * 72}")
        print(f"{'#':>3} {'买入日期':<12} {'持仓':>4} {'原始收益':>10} {'净收益':>10} {'卖出日期':<12}")
        print("-" * 72)
        for i, t in enumerate(max_hold_trades, 1):
            entry_date = dates[t['entry_idx']] if t['entry_idx'] < len(dates) else "N/A"
            exit_date = dates[t['exit_idx']] if t['exit_idx'] < len(dates) else "N/A"
            print(f"{i:>3} {entry_date:<12} {t['days_held']:>4}d {t['raw_return']:>+9.2%} {t['net_return']:>+9.2%} {exit_date:<12}")
        print(f"\n  强制平仓次数: {len(max_hold_trades)} / {len(trades)} ({len(max_hold_trades)/len(trades)*100:.1f}%)")
        
        # 强制平仓的收益统计
        mh_rets = [t['net_return'] for t in max_hold_trades]
        mh_wins = [r for r in mh_rets if r > 0]
        print(f"  强平交易胜率: {len(mh_wins)/len(max_hold_trades)*100:.1f}%")
        print(f"  强平交易平均收益: {np.mean(mh_rets):+.4f} ({np.mean(mh_rets):+.2%})")
    else:
        print(f"\n✅ 没有交易触发 max_hold_days 强制平仓")
    
    # 退出原因统计
    print(f"\n【退出原因分布】")
    reason_counts = {}
    for t in trades:
        r = t.get('exit_reason', 'unknown')
        reason_counts[r] = reason_counts.get(r, 0) + 1
    reason_names = {'signal': '信号触发卖出', 'max_hold': '强制平仓(满60天)', 'end_of_data': '数据末尾强制平'}
    for r, cnt in reason_counts.items():
        print(f"  {reason_names.get(r, r)}: {cnt} 次 ({cnt/len(trades)*100:.1f}%)")
    print(f"\n【累计收益走势 (最近20笔交易)】")
    cumsum = 0
    print(f"{'#':>3} {'日期':<12} {'持仓':>4} {'原始收益':>10} {'净收益':>10} {'累计':>10}")
    print("-" * 60)
    for i, t in enumerate(trades[-20:], 1):
        entry_date = dates[t['entry_idx']] if t['entry_idx'] < len(dates) else "N/A"
        cumsum += t['net_return']
        print(f"{i:>3} {entry_date:<12} {t['days_held']:>4}d {t['raw_return']:>+9.2%} {t['net_return']:>+9.2%} {cumsum:>+9.2%}")
    
    # 7. 信号分布
    signal = torch.sigmoid(res)[0].cpu().numpy()
    print(f"\n【Signal 分布】")
    print(f"  均值: {signal.mean():.4f}")
    print(f"  中位数: {np.median(signal):.4f}")
    for th in [0.30, 0.50, 0.70, 0.80, 0.90]:
        count = (signal > th).sum()
        print(f"  signal > {th:.2f}: {count:4d} 天 ({count/len(signal)*100:5.1f}%)")
    
    # 8. 最近信号
    print(f"\n【最近10日信号】")
    print(f"{'日期':<12} {'Signal':>8} {'状态':>10}")
    for i in range(max(0, len(signal)-10), len(signal)):
        s = signal[i]
        date = dates[i]
        if s >= 0.70:
            status = "🟢 强"
        elif s >= 0.50:
            status = "🟡 中"
        elif s >= 0.30:
            status = "🟠 弱"
        else:
            status = "⚪ 无"
        print(f"{date:<12} {s:>8.4f} {status:>10}")
    
    print(f"\n{'=' * 70}")
    print(f"回测完成")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        stock_code = sys.argv[1]
    else:
        print("用法: python run_backtest.py <股票代码>")
        print("例: python run_backtest.py 600150")
        sys.exit(1)
    
    run_backtest(stock_code)
