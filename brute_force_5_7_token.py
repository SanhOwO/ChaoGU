import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'AlphaGPT'))

import torch
import json
import time
from itertools import product
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

from model_core.data_loader import AShareDataLoader
from model_core.vm import StackVM
from model_core.backtest import AShareBacktest
from model_core.config import ModelConfig
from model_core.vocab import FORMULA_VOCAB
from model_core.ops import OPS_CONFIG

# 全局变量，供 worker 使用（在 worker_init 中设置）
_worker_feat_tensor = None
_worker_raw_data = None
_worker_target_ret = None
_worker_binary_op_ids = None
_worker_feature_ids = None

def worker_init(feat_tensor, raw_data, target_ret, binary_op_ids, feature_ids):
    """每个 worker 初始化时调用"""
    global _worker_feat_tensor, _worker_raw_data, _worker_target_ret
    global _worker_binary_op_ids, _worker_feature_ids
    _worker_feat_tensor = feat_tensor
    _worker_raw_data = raw_data
    _worker_target_ret = target_ret
    _worker_binary_op_ids = binary_op_ids
    _worker_feature_ids = feature_ids

def evaluate_formula(formula):
    """评估一个公式，返回结果或 None"""
    vm = StackVM()
    bt = AShareBacktest()
    res = vm.execute(formula, _worker_feat_tensor)
    if res is None:
        return None
    score, avg_ret = bt.evaluate(res, _worker_raw_data, _worker_target_ret)
    return {
        'formula_ids': formula,
        'score': score.item(),
        'avg_ret': avg_ret,
    }

def generate_rpn_structures(length, feature_ids, op_ids):
    """
    生成所有有效的 RPN（后缀）公式结构。
    
    RPN 有效条件：遍历序列时，栈中元素数必须始终 >= 算子需要的操作数。
    
    用递归生成：每一步选择压入一个操作数（特征）或执行一个算子。
    
    Args:
        length: 公式长度（必须是奇数）
        feature_ids: 可用特征 token id 列表
        op_ids: 可用算子 token id 列表（二元算子）
    
    Yields:
        每个有效公式是一个 token id 列表
    """
    
    def _gen(pos, stack_depth, current):
        if pos == length:
            if stack_depth == 1:
                yield current[:]
            return
        
        # 剩余位置数
        remaining = length - pos
        
        # 如果栈深度足够，可以尝试执行一个算子
        if stack_depth >= 2:
            for op_id in op_ids:
                current.append(op_id)
                yield from _gen(pos + 1, stack_depth - 1, current)  # 弹出2个，压入1个，净减1
                current.pop()
        
        # 尝试压入一个操作数（特征）
        # 需要确保剩余位置足够让栈最终归1
        # 当前 stack_depth，还剩 remaining-1 个位置（压入1个后）
        # 最大可能栈深度 = stack_depth + 1 + (remaining-1) = stack_depth + remaining
        # 但要让最终 stack_depth=1，需要足够的算子来消耗
        # 简单剪枝：如果还能压入操作数
        if stack_depth + remaining - 1 >= 1:  # 简单剪枝，确保最终能归1
            for f_id in feature_ids:
                current.append(f_id)
                yield from _gen(pos + 1, stack_depth + 1, current)
                current.pop()
    
    yield from _gen(0, 0, [])

def main():
    print("=" * 70)
    print("5-Token / 7-Token 穷举搜索（RPN，仅特征 token，不含常数）")
    print(f"训练区间: {ModelConfig.TRAIN_START_DATE} ~ {ModelConfig.TRAIN_END_DATE or 'latest'}")
    print(f"指定股票: {ModelConfig.TRAIN_SPECIFIC_CODES}")
    print("=" * 70)
    
    # 1. 加载数据
    print("\n[1/4] 加载数据...")
    loader = AShareDataLoader()
    loader.load_data(
        start_date=ModelConfig.TRAIN_START_DATE,
        end_date=ModelConfig.TRAIN_END_DATE,
        specific_codes=ModelConfig.TRAIN_SPECIFIC_CODES
    )
    print(f"  完成: {len(loader.codes)} 只股票, {len(loader.dates)} 个交易日")
    
    # 2. 确定可用 token
    feature_ids = list(range(FORMULA_VOCAB.operator_offset))  # 0-11
    binary_op_ids = []
    for i, cfg in enumerate(OPS_CONFIG):
        if cfg[2] == 2:  # arity == 2
            binary_op_ids.append(FORMULA_VOCAB.operator_offset + i)
    
    print(f"\n  二元算子: {[FORMULA_VOCAB.token_names[i] for i in binary_op_ids]} ({len(binary_op_ids)} 个)")
    print(f"  特征 token: {[FORMULA_VOCAB.token_names[i] for i in feature_ids]} ({len(feature_ids)} 个)")
    
    # 3. 生成 5-token 公式
    print("\n[2/4] 生成 5-token 公式...")
    formulas_5 = []
    for formula in generate_rpn_structures(5, feature_ids, binary_op_ids):
        formulas_5.append(formula)
    print(f"  5-token 有效结构数: {len(formulas_5)}")
    
    # 4. 生成 7-token 公式
    print("\n[3/4] 生成 7-token 公式...")
    formulas_7 = []
    for formula in generate_rpn_structures(7, feature_ids, binary_op_ids):
        formulas_7.append(formula)
    print(f"  7-token 有效结构数: {len(formulas_7)}")
    
    # 5. 多进程评估
    print(f"\n[4/4] 多进程评估（CPU 核心: {cpu_count()}）...")
    
    # 把数据序列化（简单起见，每个 worker 自己初始化）
    # 实际上使用共享对象，通过 worker_init 传递
    all_formulas = formulas_5 + formulas_7
    print(f"  总公式数: {len(all_formulas)} (5t: {len(formulas_5)}, 7t: {len(formulas_7)})")
    
    # 使用进程池
    num_workers = max(1, cpu_count() - 1)  # 留一个核心给系统
    print(f"  Worker 数: {num_workers}")
    
    start_time = time.time()
    
    with Pool(
        processes=num_workers,
        initializer=worker_init,
        initargs=(loader.feat_tensor, loader.raw_data_cache, loader.target_ret, 
                  binary_op_ids, feature_ids)
    ) as pool:
        results = []
        for i, result in enumerate(tqdm(
            pool.imap_unordered(evaluate_formula, all_formulas, chunksize=100),
            total=len(all_formulas),
            desc="评估中"
        )):
            if result is not None:
                results.append(result)
    
    elapsed = time.time() - start_time
    print(f"\n  评估完成: {len(results)}/{len(all_formulas)} 有效，耗时 {elapsed:.1f}s")
    
    # 6. 排序并输出
    results.sort(key=lambda x: x['score'], reverse=True)
    
    print(f"\n{'='*70}")
    print(f"Top 20 公式（5-token + 7-token）")
    print(f"{'='*70}")
    print(f"{'排名':<4} {'长度':<4} {'Score':>10} {'AvgRet':>10} {'公式'}")
    print(f"{'-'*70}")
    
    for rank, r in enumerate(results[:20], 1):
        formula_names = [FORMULA_VOCAB.token_names[i] for i in r['formula_ids']]
        formula_str = ' '.join(formula_names)
        length = len(r['formula_ids'])
        print(f"#{rank:<3} {length:<4} {r['score']:>+10.6f} {r['avg_ret']:>+9.2%}   {formula_str}")
    
    print(f"{'-'*70}")
    
    # 7. 按长度统计
    results_5 = [r for r in results if len(r['formula_ids']) == 5]
    results_7 = [r for r in results if len(r['formula_ids']) == 7]
    
    for label, rs in [('5-token', results_5), ('7-token', results_7)]:
        if not rs:
            continue
        positive = len([r for r in rs if r['score'] > 0])
        negative = len([r for r in rs if r['score'] < 0])
        zero = len([r for r in rs if r['score'] == 0])
        print(f"\n{label} 统计:")
        print(f"  有效: {len(rs)} | 正收益: {positive} | 负收益: {negative} | 零收益: {zero}")
        if rs:
            print(f"  最优: {' '.join([FORMULA_VOCAB.token_names[i] for i in rs[0]['formula_ids']])} (Score={rs[0]['score']:+.6f})")
    
    # 8. 保存结果
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'brute_force_5_7_token_results.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        # 转换公式为名称以便可读
        output = []
        for r in results:
            r_copy = r.copy()
            r_copy['formula_names'] = [FORMULA_VOCAB.token_names[i] for i in r['formula_ids']]
            output.append(r_copy)
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存: {output_path}")
    
    return results

if __name__ == '__main__':
    # Windows 多进程需要
    import multiprocessing
    multiprocessing.set_start_method('spawn', force=True)
    main()
