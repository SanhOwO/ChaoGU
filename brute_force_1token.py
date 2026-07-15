import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'AlphaGPT'))

import torch
from model_core.data_loader import AShareDataLoader
from model_core.vm import StackVM
from model_core.backtest import AShareBacktest
from model_core.config import ModelConfig
from model_core.vocab import FORMULA_VOCAB

def main():
    print("=" * 60)
    print("1-Token 穷举搜索")
    print(f"训练区间: {ModelConfig.TRAIN_START_DATE} ~ {ModelConfig.TRAIN_END_DATE or 'latest'}")
    print(f"指定股票: {ModelConfig.TRAIN_SPECIFIC_CODES}")
    print("=" * 60)
    
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
    
    # 3. 穷举所有 1-token 公式
    results = []
    vocab_size = FORMULA_VOCAB.size
    print(f"\n穷举 {vocab_size} 个 token...")
    
    for token_id in range(vocab_size):
        token_name = FORMULA_VOCAB.token_names[token_id]
        
        # 执行 1-token 公式
        res = vm.execute([token_id], loader.feat_tensor)
        
        if res is None:
            # 算子 token 需要操作数，1-token 执行失败
            continue
        
        # 回测评估
        score, avg_ret = bt.evaluate(res, loader.raw_data_cache, loader.target_ret)
        
        results.append({
            'token_id': token_id,
            'token_name': token_name,
            'score': score.item(),
            'avg_ret': avg_ret,
            'category': '特征' if token_id < FORMULA_VOCAB.operator_offset else '常数'
        })
    
    # 4. 排序并输出
    results.sort(key=lambda x: x['score'], reverse=True)
    
    print(f"\n有效 1-token 公式: {len(results)} / {vocab_size}")
    print("-" * 60)
    print(f"{'排名':<4} {'TokenID':<8} {'名称':<15} {'类别':<6} {'Score':>10} {'AvgRet':>10}")
    print("-" * 60)
    
    for rank, r in enumerate(results, 1):
        print(f"#{rank:<3} {r['token_id']:<8} {r['token_name']:<15} {r['category']:<6} {r['score']:>+10.6f} {r['avg_ret']:>+9.2%}")
    
    print("-" * 60)
    
    # 5. 保存结果到文件
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'brute_force_1token_results.json')
    import json
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存: {output_path}")
    
    # 6. 统计
    positive = [r for r in results if r['score'] > 0]
    negative = [r for r in results if r['score'] < 0]
    zero = [r for r in results if r['score'] == 0]
    
    print(f"\n统计:")
    print(f"  正收益: {len(positive)} 个")
    print(f"  负收益: {len(negative)} 个")
    print(f"  零收益: {len(zero)} 个")
    print(f"  最优: {results[0]['token_name']} (Score={results[0]['score']:+.6f})")
    if len(results) > 1:
        print(f"  次优: {results[1]['token_name']} (Score={results[1]['score']:+.6f})")
    
    return results

if __name__ == '__main__':
    main()
