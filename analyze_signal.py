#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
信号分布分析脚本
用法: python analyze_signal.py <股票代码>
例: python analyze_signal.py 002548

功能：
1. 加载训练好的策略公式
2. 回算历史信号分布（均值、分位数、阈值触发频率）
3. 按信号强度分桶，统计各桶的后续收益与胜率
4. 输出最近交易日的信号值（用于实盘参考）
"""

import sys
import os
import json
import numpy as np

# 把 AlphaGPT 加入路径
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'AlphaGPT'))

from model_core.data_loader import AShareDataLoader
from model_core.vm import StackVM
from model_core.config import ModelConfig
from model_core.vocab import FORMULA_VOCAB
import torch


def load_strategy(stock_code):
    """加载策略文件"""
    strategy_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "个股训练数据",
        f"{stock_code}_strategy.json"
    )
    if not os.path.exists(strategy_path):
        raise FileNotFoundError(f"策略文件不存在: {strategy_path}\n请先运行训练: python train_single.py {stock_code}")
    
    with open(strategy_path, "r") as f:
        formula_tokens = json.load(f)
    
    print(f"[OK] 加载策略: {strategy_path}")
    print(f"     公式 Token: {formula_tokens}")
    
    # 翻译 token 为可读名称
    token_names = FORMULA_VOCAB.token_names
    readable = []
    for t in formula_tokens:
        if t < len(token_names):
            readable.append(token_names[t])
        else:
            readable.append(f"UNK({t})")
    print(f"     公式解读: {' | '.join(readable)}")
    
    return formula_tokens


def load_stock_data(stock_code):
    """加载单只个股数据"""
    loader = AShareDataLoader()
    loader.load_data(
        start_date=ModelConfig.TRAIN_START_DATE,
        end_date=ModelConfig.TRAIN_END_DATE,
        specific_codes=[stock_code]
    )
    return loader


def compute_signal(formula_tokens, loader):
    """执行公式，计算 sigmoid 信号"""
    vm = StackVM()
    
    # 尝试不同前缀长度（和训练时一致）
    res = None
    for trunc_len in [5, 7, 9, 11, 13]:
        if trunc_len > len(formula_tokens):
            continue
        candidate = formula_tokens[:trunc_len]
        res = vm.execute(candidate, loader.feat_tensor)
        if res is not None:
            break
    
    if res is None:
        raise RuntimeError("公式执行失败，所有前缀长度均无效。策略文件可能损坏。")
    
    # sigmoid 压缩到 0~1
    signal = torch.sigmoid(res)
    
    # 取单股（Stocks=1）
    signal_1d = signal[0].cpu().numpy()
    
    return signal_1d, res[0].cpu().numpy(), candidate


def analyze_distribution(signal, dates, stock_code):
    """信号分布统计"""
    print("\n" + "=" * 60)
    print(f"📊 {stock_code} 信号分布分析")
    print("=" * 60)
    
    # 基础统计
    print(f"\n【基础统计】")
    print(f"  数据天数: {len(signal)}")
    print(f"  信号均值: {signal.mean():.4f}")
    print(f"  信号中位数: {np.median(signal):.4f}")
    print(f"  标准差: {signal.std():.4f}")
    print(f"  最小值: {signal.min():.4f}")
    print(f"  最大值: {signal.max():.4f}")
    
    # 分位数
    print(f"\n【信号分位数】")
    for q in [10, 20, 30, 40, 50, 60, 70, 80, 90]:
        val = np.percentile(signal, q)
        print(f"  P{q:2d}: {val:.4f}")
    
    # 阈值触发频率
    print(f"\n【阈值触发频率】")
    thresholds = [0.30, 0.40, 0.50, 0.55, 0.60, 0.70, 0.80, 0.90]
    for th in thresholds:
        count = (signal > th).sum()
        pct = count / len(signal) * 100
        print(f"  signal > {th:.2f}: {count:4d} 天 ({pct:5.1f}%)")


def analyze_buckets(signal, target_ret, dates, stock_code):
    """按信号强度分桶，分析后续收益"""
    target = target_ret[0].cpu().numpy()  # 单股
    
    # 只取有有效目标收益的天数（最后两天为0，因无法计算）
    valid_mask = np.abs(target) > 1e-10
    s_valid = signal[valid_mask]
    r_valid = target[valid_mask]
    d_valid = [dates[i] for i in range(len(valid_mask)) if valid_mask[i]]
    
    print(f"\n【信号分桶收益分析】")
    print(f"  有效交易日: {len(s_valid)} 天")
    print(f"  (T日信号 → T+1开盘买入, T+2开盘卖出, 已含交易成本)")
    print()
    
    # 定义桶
    buckets = [
        ("signal < 0.30", lambda s: s < 0.30),
        ("0.30 ≤ signal < 0.50", lambda s: (s >= 0.30) & (s < 0.50)),
        ("0.50 ≤ signal < 0.70", lambda s: (s >= 0.50) & (s < 0.70)),
        ("signal ≥ 0.70", lambda s: s >= 0.70),
    ]
    
    print(f"{'分桶':<22} {'天数':>6} {'平均收益':>10} {'胜率':>8} {'最大盈利':>10} {'最大亏损':>10}")
    print("-" * 75)
    
    for name, mask_fn in buckets:
        mask = mask_fn(s_valid)
        count = mask.sum()
        if count == 0:
            print(f"{name:<22} {count:>6} {'N/A':>10} {'N/A':>8} {'N/A':>10} {'N/A':>10}")
            continue
        
        rets = r_valid[mask]
        avg_ret = rets.mean()
        win_rate = (rets > 0).sum() / count * 100
        max_gain = rets.max()
        max_loss = rets.min()
        
        print(f"{name:<22} {count:>6} {avg_ret:>+9.2%} {win_rate:>7.1f}% {max_gain:>+9.2%} {max_loss:>+9.2%}")
    
    # 额外：打印信号与收益的相关性
    if len(s_valid) > 10:
        corr = np.corrcoef(s_valid, r_valid)[0, 1]
        print(f"\n  信号-收益相关性: {corr:+.4f}")
        if corr > 0.1:
            print(f"  ✅ 信号与收益呈正相关，公式有一定预测能力")
        elif corr < -0.1:
            print(f"  ⚠️  信号与收益呈负相关，考虑反向使用")
        else:
            print(f"  ⚠️  信号与收益相关性弱，公式区分度有限")


def print_recent_signals(signal, dates, stock_code, n_days=20):
    """打印最近 N 天的信号"""
    print(f"\n【最近 {n_days} 日信号明细】")
    print(f"{'日期':<12} {'Signal':>8} {'状态':>10}")
    print("-" * 35)
    
    for i in range(max(0, len(signal) - n_days), len(signal)):
        s = signal[i]
        date = dates[i]
        if s >= 0.70:
            status = "🟢 强信号"
        elif s >= 0.50:
            status = "🟡 中信号"
        elif s >= 0.30:
            status = "🟠 弱信号"
        else:
            status = "⚪ 无信号"
        print(f"{date:<12} {s:>8.4f} {status:>10}")
    
    # 最新一天的特别标注
    latest_signal = signal[-1]
    print(f"\n💡 最新信号 ({dates[-1]}): {latest_signal:.4f}")
    if latest_signal >= 0.70:
        print(f"   → 达到买入阈值 (≥0.70)，次日开盘可考虑买入")
    elif latest_signal >= 0.50:
        print(f"   → 中等强度信号，建议观察")
    else:
        print(f"   → 未达买入阈值，空仓/观望")


def main():
    # 解析命令行参数
    if len(sys.argv) > 1:
        stock_code = sys.argv[1]
    else:
        print("用法: python analyze_signal.py <股票代码>")
        print("例: python analyze_signal.py 002548")
        sys.exit(1)
    
    print(f"\n>>> 分析股票: {stock_code}")
    
    # 1. 加载策略
    formula_tokens = load_strategy(stock_code)
    
    # 2. 加载数据
    print(f"\n[...] 加载数据 ({ModelConfig.TRAIN_START_DATE or '最早'} ~ {ModelConfig.TRAIN_END_DATE or '最新'})...")
    loader = load_stock_data(stock_code)
    dates = loader.dates
    print(f"[OK] 数据加载完成: {len(dates)} 个交易日")
    
    # 3. 计算信号
    signal, raw_signal, used_tokens = compute_signal(formula_tokens, loader)
    print(f"\n[OK] 公式执行成功 (使用前缀长度: {len(used_tokens)})")
    
    # 4. 分布分析
    analyze_distribution(signal, dates, stock_code)
    
    # 5. 分桶收益分析
    analyze_buckets(signal, loader.target_ret, dates, stock_code)
    
    # 6. 最近信号
    print_recent_signals(signal, dates, stock_code, n_days=20)
    
    print("\n" + "=" * 60)
    print("分析完成")
    print("=" * 60)


if __name__ == "__main__":
    main()
