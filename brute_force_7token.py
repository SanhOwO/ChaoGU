import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'AlphaGPT'))

import torch
import json
import time
import multiprocessing as mp
from multiprocessing import Pool
from itertools import product
from tqdm import tqdm

from model_core.data_loader import AShareDataLoader
from model_core.vm import StackVM
from model_core.backtest import AShareBacktest
from model_core.config import ModelConfig
from model_core.vocab import FORMULA_VOCAB
from model_core.ops import OPS_CONFIG


def worker_task(args):
    """
    每个 worker 处理一个子任务：固定结构 + 固定第一个特征 a
    
    5 种 RPN 结构（4 个操作数 + 3 个二元算子）：
    0: A B OP1 C D OP3 OP2  → (A OP1 B) OP2 (C OP3 D)
    1: A B OP1 C OP2 D OP3  → ((A OP1 B) OP2 C) OP3 D
    2: A B C OP2 OP1 D OP3  → (A OP1 (B OP2 C)) OP3 D
    3: A B C OP2 D OP3 OP1  → A OP1 ((B OP2 C) OP3 D)
    4: A B C D OP3 OP2 OP1  → A OP1 (B OP2 (C OP3 D))
    """
    structure_id, a_val, feature_ids, op_ids, start_date, end_date, specific_codes = args
    
    # 加载数据（每个 worker 独立加载）
    loader = AShareDataLoader()
    loader.load_data(
        start_date=start_date,
        end_date=end_date,
        specific_codes=specific_codes
    )
    vm = StackVM()
    bt = AShareBacktest()
    
    results = []
    total = len(op_ids)**3 * len(feature_ids)**3
    count = 0
    
    for op1, op2, op3 in product(op_ids, repeat=3):
        for b in feature_ids:
            for c in feature_ids:
                for d in feature_ids:
                    count += 1
                    
                    # 根据结构 ID 生成公式
                    if structure_id == 0:
                        formula = [a_val, b, op1, c, d, op3, op2]
                    elif structure_id == 1:
                        formula = [a_val, b, op1, c, op2, d, op3]
                    elif structure_id == 2:
                        formula = [a_val, b, c, op2, op1, d, op3]
                    elif structure_id == 3:
                        formula = [a_val, b, c, op2, d, op3, op1]
                    elif structure_id == 4:
                        formula = [a_val, b, c, d, op3, op2, op1]
                    
                    # 评估
                    res = vm.execute(formula, loader.feat_tensor)
                    if res is None:
                        continue
                    
                    score, avg_ret = bt.evaluate(res, loader.raw_data_cache, loader.target_ret)
                    results.append({
                        'formula_ids': formula,
                        'score': score.item(),
                        'avg_ret': avg_ret,
                    })
    
    return results


def main():
    print("=" * 70)
    print("7-Token 穷举搜索（RPN，仅特征 token + 二元算子，多进程）")
    print(f"训练区间: {ModelConfig.TRAIN_START_DATE} ~ {ModelConfig.TRAIN_END_DATE or 'latest'}")
    print(f"指定股票: {ModelConfig.TRAIN_SPECIFIC_CODES}")
    print("=" * 70)
    
    # 1. 确定可用 token
    feature_ids = list(range(FORMULA_VOCAB.operator_offset))  # 0-11
    binary_op_ids = []
    for i, cfg in enumerate(OPS_CONFIG):
        if cfg[2] == 2:  # arity == 2
            binary_op_ids.append(FORMULA_VOCAB.operator_offset + i)
    
    print(f"\n  二元算子: {[FORMULA_VOCAB.token_names[i] for i in binary_op_ids]} ({len(binary_op_ids)} 个)")
    print(f"  特征 token: {[FORMULA_VOCAB.token_names[i] for i in feature_ids]} ({len(feature_ids)} 个)")
    
    # 5 种结构 × 12 个 a 值 = 60 个子任务
    num_structures = 5
    num_a = len(feature_ids)
    total_subtasks = num_structures * num_a
    total_formulas = num_structures * len(feature_ids)**4 * len(binary_op_ids)**3
    print(f"\n  7-token 有效结构: {num_structures} 种")
    print(f"  总组合数: {total_formulas:,}")
    print(f"  子任务数: {total_subtasks} (每个约 {total_formulas // total_subtasks:,} 种组合)")
    
    # 2. 准备任务列表
    tasks = []
    for structure_id in range(num_structures):
        for a_val in feature_ids:
            tasks.append((
                structure_id, a_val, feature_ids, binary_op_ids,
                ModelConfig.TRAIN_START_DATE, ModelConfig.TRAIN_END_DATE,
                ModelConfig.TRAIN_SPECIFIC_CODES
            ))
    
    # 3. 多进程评估
    num_workers = max(1, mp.cpu_count() - 1)
    print(f"\n[1/2] 多进程评估（Worker: {num_workers}）...")
    print(f"  预计时间: 取决于 CPU 性能，参考 5-token 耗时按比例估算")
    
    start_time = time.time()
    all_results = []
    
    with Pool(processes=num_workers) as pool:
        for batch_results in tqdm(
            pool.imap_unordered(worker_task, tasks),
            total=len(tasks),
            desc="评估"
        ):
            all_results.extend(batch_results)
    
    elapsed = time.time() - start_time
    print(f"\n  完成: {len(all_results)}/{total_formulas} 有效，耗时 {elapsed:.1f}s")
    
    # 4. 排序并输出
    all_results.sort(key=lambda x: x['score'], reverse=True)
    
    print(f"\n{'='*70}")
    print(f"Top 20 7-Token 公式")
    print(f"{'='*70}")
    print(f"{'排名':<4} {'Score':>10} {'AvgRet':>10} {'公式'}")
    print(f"{'-'*70}")
    
    for rank, r in enumerate(all_results[:20], 1):
        formula_names = [FORMULA_VOCAB.token_names[i] for i in r['formula_ids']]
        formula_str = ' '.join(formula_names)
        print(f"#{rank:<3} {r['score']:>+10.6f} {r['avg_ret']:>+9.2%}   {formula_str}")
    
    print(f"{'-'*70}")
    
    # 5. 统计
    positive = len([r for r in all_results if r['score'] > 0])
    negative = len([r for r in all_results if r['score'] < 0])
    zero = len([r for r in all_results if r['score'] == 0])
    total = len(all_results)
    
    print(f"\n统计:")
    print(f"  总有效: {total:,}")
    print(f"  正收益: {positive:,} ({positive/total*100:.1f}%)")
    print(f"  负收益: {negative:,} ({negative/total*100:.1f}%)")
    print(f"  零收益: {zero:,} ({zero/total*100:.1f}%)")
    if all_results:
        best_names = [FORMULA_VOCAB.token_names[i] for i in all_results[0]['formula_ids']]
        print(f"  最优: {' '.join(best_names)} (Score={all_results[0]['score']:+.6f})")
    
    # 6. 保存结果
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'brute_force_7token_results.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        output = []
        for r in all_results:
            r_copy = r.copy()
            r_copy['formula_names'] = [FORMULA_VOCAB.token_names[i] for i in r['formula_ids']]
            output.append(r_copy)
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存: {output_path}")
    
    return all_results


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    main()
