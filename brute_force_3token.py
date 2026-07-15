import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'AlphaGPT'))

import torch
from itertools import product
from model_core.data_loader import AShareDataLoader
from model_core.vm import StackVM
from model_core.backtest import AShareBacktest
from model_core.config import ModelConfig
from model_core.vocab import FORMULA_VOCAB
from model_core.ops import OPS_CONFIG

def main():
    print("=" * 70)
    print("3-Token 穷举搜索（仅特征 token，不含常数）")
    print(f"训练区间: {ModelConfig.TRAIN_START_DATE} ~ {ModelConfig.TRAIN_END_DATE or 'latest'}")
    print(f"指定股票: {ModelConfig.TRAIN_SPECIFIC_CODES}")
    print("=" * 70)
    
    # 1. 加载数据
    loader = AShareDataLoader()
    loader.load_data(
        start_date=ModelConfig.TRAIN_START_DATE,
        end_date=ModelConfig.TRAIN_END_DATE,
        specific_codes=ModelConfig.TRAIN_SPECIFIC_CODES
    )
    print(f"  加载完成: {len(loader.codes)} 只股票, {len(loader.dates)} 个交易日")
    
    # 2. 初始化 VM 和回测
    vm = StackVM()
    bt = AShareBacktest()
    
    # 3. 确定二元算子和特征 token
    feature_ids = list(range(FORMULA_VOCAB.operator_offset))  # 0-11
    binary_op_ids = []
    for i, cfg in enumerate(OPS_CONFIG):
        if cfg[2] == 2:  # arity == 2
            binary_op_ids.append(FORMULA_VOCAB.operator_offset + i)
    
    print(f"\n二元算子: {[FORMULA_VOCAB.token_names[i] for i in binary_op_ids]}")
    print(f"特征 token: {[FORMULA_VOCAB.token_names[i] for i in feature_ids]}")
    print(f"总组合数: {len(binary_op_ids)} × {len(feature_ids)} × {len(feature_ids)} = {len(binary_op_ids) * len(feature_ids) * len(feature_ids)}")
    
    # 4. 穷举所有 3-token 公式: 算子 操作数1 操作数2
    results = []
    total = len(binary_op_ids) * len(feature_ids) * len(feature_ids)
    count = 0
    
    for op_id in binary_op_ids:
        op_name = FORMULA_VOCAB.token_names[op_id]
        for f1_id in feature_ids:
            f1_name = FORMULA_VOCAB.token_names[f1_id]
            for f2_id in feature_ids:
                f2_name = FORMULA_VOCAB.token_names[f2_id]
                count += 1
                
                formula = [f1_id, f2_id, op_id]
                
                # VM 执行 (RPN: 操作数在前，算子在后)
                res = vm.execute(formula, loader.feat_tensor)
                if res is None:
                    continue
                
                # 回测评估
                score, avg_ret = bt.evaluate(res, loader.raw_data_cache, loader.target_ret)
                
                results.append({
                    'formula': [op_name, f1_name, f2_name],
                    'formula_ids': [op_id, f1_id, f2_id],
                    'score': score.item(),
                    'avg_ret': avg_ret,
                })
                
                if count % 100 == 0 or count == total:
                    print(f"  进度: {count}/{total} | 有效: {len(results)} | 最优: {max(r['score'] for r in results) if results else 'N/A':.6f}", end='\r')
    
    print()  # newline
    
    # 5. 排序并输出
    results.sort(key=lambda x: x['score'], reverse=True)
    
    print(f"\n有效 3-token 公式: {len(results)} / {total}")
    print("-" * 70)
    print(f"{'排名':<4} {'Score':>10} {'AvgRet':>10} {'公式'}")
    print("-" * 70)
    
    for rank, r in enumerate(results[:20], 1):
        formula_str = ' '.join(r['formula'])
        print(f"#{rank:<3} {r['score']:>+10.6f} {r['avg_ret']:>+9.2%}   {formula_str}")
    
    if len(results) > 20:
        print(f"  ... 共 {len(results)} 个，显示前 20")
    
    print("-" * 70)
    
    # 6. 保存结果
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'brute_force_3token_results.json')
    import json
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存: {output_path}")
    
    # 7. 统计
    positive = [r for r in results if r['score'] > 0]
    negative = [r for r in results if r['score'] < 0]
    zero = [r for r in results if r['score'] == 0]
    
    print(f"\n统计:")
    print(f"  正收益: {len(positive)} 个 ({len(positive)/len(results)*100:.1f}%)")
    print(f"  负收益: {len(negative)} 个 ({len(negative)/len(results)*100:.1f}%)")
    print(f"  零收益: {len(zero)} 个")
    print(f"  最优: {' '.join(results[0]['formula'])} (Score={results[0]['score']:+.6f})")
    if len(results) > 1:
        print(f"  次优: {' '.join(results[1]['formula'])} (Score={results[1]['score']:+.6f})")
    
    return results

if __name__ == '__main__':
    main()
