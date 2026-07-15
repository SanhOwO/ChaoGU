import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'AlphaGPT'))

import torch
import json
import time
from itertools import product
from tqdm import tqdm

from model_core.data_loader import AShareDataLoader
from model_core.vm import StackVM
from model_core.backtest import AShareBacktest
from model_core.config import ModelConfig
from model_core.vocab import FORMULA_VOCAB
from model_core.ops import OPS_CONFIG

def generate_rpn_5token(feature_ids, op_ids):
    """
    生成所有有效的 5-token RPN 公式。
    5-token 只有 2 种二叉树结构（Catalan(2)=2）：
    1. [A, B, OP1, C, OP2]  -> 先算 A OP1 B，再算结果 OP2 C
    2. [A, B, C, OP1, OP2]  -> 先算 B OP1 C，再算 A OP2 结果
    """
    for op1 in op_ids:
        for op2 in op_ids:
            for a in feature_ids:
                for b in feature_ids:
                    for c in feature_ids:
                        yield [a, b, op1, c, op2]  # 结构1
                        yield [a, b, c, op1, op2]   # 结构2

def main():
    print("=" * 70)
    print("5-Token 穷举搜索（RPN，仅特征 token，单线程）")
    print(f"训练区间: {ModelConfig.TRAIN_START_DATE} ~ {ModelConfig.TRAIN_END_DATE or 'latest'}")
    print(f"指定股票: {ModelConfig.TRAIN_SPECIFIC_CODES}")
    print("=" * 70)
    
    # 1. 加载数据
    print("\n[1/3] 加载数据...")
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
    
    # 3. 穷举所有 5-token 公式
    total = 2 * len(binary_op_ids)**2 * len(feature_ids)**3
    print(f"\n[2/3] 穷举评估... 总组合数: {total:,}")
    
    vm = StackVM()
    bt = AShareBacktest()
    results = []
    start_time = time.time()
    
    for formula in tqdm(generate_rpn_5token(feature_ids, binary_op_ids), total=total, desc="评估"):
        res = vm.execute(formula, loader.feat_tensor)
        if res is None:
            continue
        score, avg_ret = bt.evaluate(res, loader.raw_data_cache, loader.target_ret)
        results.append({
            'formula_ids': formula,
            'score': score.item(),
            'avg_ret': avg_ret,
        })
    
    elapsed = time.time() - start_time
    print(f"\n  完成: {len(results)}/{total} 有效，耗时 {elapsed:.1f}s")
    
    # 4. 排序并输出
    results.sort(key=lambda x: x['score'], reverse=True)
    
    print(f"\n{'='*70}")
    print(f"Top 20 5-Token 公式")
    print(f"{'='*70}")
    print(f"{'排名':<4} {'Score':>10} {'AvgRet':>10} {'公式'}")
    print(f"{'-'*70}")
    
    for rank, r in enumerate(results[:20], 1):
        formula_names = [FORMULA_VOCAB.token_names[i] for i in r['formula_ids']]
        formula_str = ' '.join(formula_names)
        print(f"#{rank:<3} {r['score']:>+10.6f} {r['avg_ret']:>+9.2%}   {formula_str}")
    
    print(f"{'-'*70}")
    
    # 5. 统计
    positive = len([r for r in results if r['score'] > 0])
    negative = len([r for r in results if r['score'] < 0])
    zero = len([r for r in results if r['score'] == 0])
    
    print(f"\n统计:")
    print(f"  正收益: {positive} ({positive/len(results)*100:.1f}%)")
    print(f"  负收益: {negative} ({negative/len(results)*100:.1f}%)")
    print(f"  零收益: {zero}")
    if results:
        print(f"  最优: {' '.join([FORMULA_VOCAB.token_names[i] for i in results[0]['formula_ids']])} (Score={results[0]['score']:+.6f})")
    
    # 6. 保存结果
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'brute_force_5token_results.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        output = []
        for r in results:
            r_copy = r.copy()
            r_copy['formula_names'] = [FORMULA_VOCAB.token_names[i] for i in r['formula_ids']]
            output.append(r_copy)
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存: {output_path}")
    
    return results

if __name__ == '__main__':
    main()
