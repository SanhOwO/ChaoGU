#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
策略归档脚本
用法: python archive_strategy.py <股票代码> [备注]
功能: 将训练好的策略公式、回测结果归档到 AlphaGPT/数据/实战/ 目录
"""

import sys
import os
import json
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'AlphaGPT'))

ARCHIVE_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "AlphaGPT", "数据", "实战"
)
os.makedirs(ARCHIVE_DIR, exist_ok=True)

ARCHIVE_FILE = os.path.join(ARCHIVE_DIR, "strategy_archive.json")


def load_archive():
    """加载现有档案"""
    if os.path.exists(ARCHIVE_FILE):
        with open(ARCHIVE_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {
        "version": "1.0",
        "description": "AlphaGPT 实战策略档案库",
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "updated_at": None,
        "strategies": []
    }


def save_archive(data):
    """保存档案"""
    data["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(ARCHIVE_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"[OK] 档案已保存: {ARCHIVE_FILE}")


def archive_strategy(stock_code, note=""):
    """归档一只股票策略"""
    from model_core.vocab import FORMULA_VOCAB
    
    # 1. 加载策略
    strategy_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "个股训练数据",
        f"{stock_code}_strategy.json"
    )
    history_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "个股训练数据",
        f"{stock_code}_training_history.json"
    )
    
    if not os.path.exists(strategy_path):
        print(f"[ERROR] 策略文件不存在: {strategy_path}")
        return False
    
    with open(strategy_path, 'r') as f:
        formula_tokens = json.load(f)
    
    # 2. 读取训练历史（获取best_score和步数）
    best_score = None
    train_steps = None
    if os.path.exists(history_path):
        with open(history_path, 'r') as f:
            history = json.load(f)
        if 'best_score' in history and history['best_score']:
            best_score = float(history['best_score'][-1])
            train_steps = len(history['step'])
    
    # 3. 翻译公式
    token_names = FORMULA_VOCAB.token_names
    readable = []
    for t in formula_tokens:
        if t < len(token_names):
            readable.append(token_names[t])
        else:
            readable.append(f"UNK({t})")
    
    # 4. 加载档案
    archive = load_archive()
    
    # 5. 检查是否已存在，存在则更新
    existing = None
    for i, s in enumerate(archive['strategies']):
        if s['stock_code'] == stock_code:
            existing = i
            break
    
    strategy_record = {
        "stock_code": stock_code,
        "formula_tokens": formula_tokens,
        "formula_readable": " | ".join(readable),
        "best_score": best_score,
        "train_steps": train_steps,
        "note": note,
        "archived_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    if existing is not None:
        old = archive['strategies'][existing]
        archive['strategies'][existing] = strategy_record
        print(f"[OK] 更新已有策略: {stock_code}")
        print(f"     旧记录时间: {old.get('archived_at', 'N/A')}")
        print(f"     新记录时间: {strategy_record['archived_at']}")
    else:
        archive['strategies'].append(strategy_record)
        print(f"[OK] 新增策略: {stock_code}")
    
    print(f"     公式: {' | '.join(readable)}")
    print(f"     Best Score: {best_score}")
    print(f"     训练步数: {train_steps}")
    if note:
        print(f"     备注: {note}")
    
    # 6. 保存
    save_archive(archive)
    print(f"     档案中共有 {len(archive['strategies'])} 只策略")
    
    return True


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python archive_strategy.py <股票代码> [备注]")
        print("例: python archive_strategy.py 600150 '训练门槛0.5，max_hold=60'")
        sys.exit(1)
    
    stock_code = sys.argv[1]
    note = sys.argv[2] if len(sys.argv) > 2 else ""
    
    archive_strategy(stock_code, note)
