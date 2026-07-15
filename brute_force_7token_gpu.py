import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'AlphaGPT'))

import torch
import json
import time
from itertools import product
from tqdm import tqdm

from model_core.data_loader import AShareDataLoader
from model_core.config import ModelConfig
from model_core.vocab import FORMULA_VOCAB
from model_core.ops import OPS_CONFIG


# 二元算子函数映射
def _add(x, y): return x + y
def _sub(x, y): return x - y
def _mul(x, y): return x * y
def _div(x, y): return x / (y + 1e-6)

OP_FUNCS = {
    'ADD': _add,
    'SUB': _sub,
    'MUL': _mul,
    'DIV': _div,
}


def batch_evaluate(factors_batch, raw_data, target_ret, device):
    """
    批量回测评估（GPU 优化版）
    
    Args:
        factors_batch: [NumFormulas, Stocks, Time]
        raw_data: dict with tensors of shape [Stocks, Time]
        target_ret: [Stocks, Time]
        device: torch device
    
    Returns:
        final_fitness: [NumFormulas] 中位数适应度
        avg_ret: [NumFormulas] 平均收益
    """
    # 1. 信号 → 买入概率
    signal = torch.sigmoid(factors_batch)  # [NumFormulas, Stocks, Time]
    
    # 2. 基础过滤（broadcast 到 batch 维度）
    close = raw_data['close'].unsqueeze(0).to(device)  # [1, Stocks, Time]
    mc = raw_data.get('market_cap', torch.ones_like(raw_data['close']) * 1e10).unsqueeze(0).to(device)
    volume = raw_data['volume'].unsqueeze(0).to(device)
    
    is_valid = (
        (close > 2.0) & (close < 500.0) &
        (mc > 1e8) &
        (volume > 0)
    ).float()
    
    # 3. 涨跌停过滤
    is_limit_up = raw_data.get('is_limit_up', torch.zeros_like(raw_data['close'])).unsqueeze(0).to(device)
    can_buy = is_valid * (1 - is_limit_up) * (signal > 0.70).float()
    
    # 4. 收益计算（broadcast target_ret）
    target_ret = target_ret.unsqueeze(0).to(device)  # [1, Stocks, Time]
    gross_pnl = can_buy * target_ret
    
    # 5. 交易成本（双边）
    buy_cost_rate = 0.00011
    sell_cost_rate = 0.00061
    transfer_fee = 0.00001
    slippage = 0.001
    total_cost = buy_cost_rate + sell_cost_rate + transfer_fee + slippage * 2
    net_pnl = gross_pnl - can_buy * total_cost
    
    # 6. 累积收益（对 Time 维度求和）
    cum_ret = net_pnl.sum(dim=2)  # [NumFormulas, Stocks]
    
    # 7. 中位数适应度（每只股票独立）
    score = cum_ret  # [NumFormulas, Stocks]
    final_fitness = torch.median(score, dim=1).values  # [NumFormulas]
    avg_ret = score.mean(dim=1)  # [NumFormulas]
    
    return final_fitness, avg_ret


def brute_force_structure_gpu(structure_id, feat_tensor, raw_data, target_ret, 
                               feature_ids, binary_op_names, device, chunk_size=4096):
    """
    在 GPU 上批量穷举一个 7-token 结构
    
    5 种 RPN 结构：
    0: [A, B, OP1, C, D, OP3, OP2] → (A OP1 B) OP2 (C OP3 D)
    1: [A, B, OP1, C, OP2, D, OP3] → ((A OP1 B) OP2 C) OP3 D
    2: [A, B, C, OP2, OP1, D, OP3] → (A OP1 (B OP2 C)) OP3 D
    3: [A, B, C, OP2, D, OP3, OP1] → A OP1 ((B OP2 C) OP3 D)
    4: [A, B, C, D, OP3, OP2, OP1] → A OP1 (B OP2 (C OP3 D))
    """
    results = []
    num_features = len(feature_ids)
    num_ops = len(binary_op_names)
    
    # 预生成所有特征组合 (A, B, C, D)
    all_combos = torch.tensor(list(product(feature_ids, repeat=4)), device=device)  # [N, 4]
    N = len(all_combos)
    
    # 特征提取：从 feat_tensor [Stocks, Features, Time] 提取
    # 结果 shape: [N, Stocks, Time]
    A = feat_tensor[:, all_combos[:, 0], :].permute(1, 0, 2)
    B = feat_tensor[:, all_combos[:, 1], :].permute(1, 0, 2)
    C = feat_tensor[:, all_combos[:, 2], :].permute(1, 0, 2)
    D = feat_tensor[:, all_combos[:, 3], :].permute(1, 0, 2)
    
    # 遍历所有算子组合 (OP1, OP2, OP3)
    total_op_combos = num_ops ** 3
    op_combo_idx = 0
    
    for op1_name in binary_op_names:
        for op2_name in binary_op_names:
            for op3_name in binary_op_names:
                op_combo_idx += 1
                op1 = OP_FUNCS[op1_name]
                op2 = OP_FUNCS[op2_name]
                op3 = OP_FUNCS[op3_name]
                
                # 按结构执行批量计算
                if structure_id == 0:
                    # (A OP1 B) OP2 (C OP3 D)
                    t1 = op1(A, B)
                    t2 = op3(C, D)
                    result = op2(t1, t2)
                elif structure_id == 1:
                    # ((A OP1 B) OP2 C) OP3 D
                    t1 = op1(A, B)
                    t2 = op2(t1, C)
                    result = op3(t2, D)
                elif structure_id == 2:
                    # (A OP1 (B OP2 C)) OP3 D
                    t1 = op2(B, C)
                    t2 = op1(A, t1)
                    result = op3(t2, D)
                elif structure_id == 3:
                    # A OP1 ((B OP2 C) OP3 D)
                    t1 = op2(B, C)
                    t2 = op3(t1, D)
                    result = op1(A, t2)
                elif structure_id == 4:
                    # A OP1 (B OP2 (C OP3 D))
                    t1 = op3(C, D)
                    t2 = op2(B, t1)
                    result = op1(A, t2)
                
                # 批量回测（分块处理，避免显存溢出）
                if N > chunk_size:
                    all_scores = []
                    all_avgs = []
                    for i in range(0, N, chunk_size):
                        end = min(i + chunk_size, N)
                        chunk_result = result[i:end]
                        scores, avgs = batch_evaluate(chunk_result, raw_data, target_ret, device)
                        all_scores.append(scores)
                        all_avgs.append(avgs)
                    scores = torch.cat(all_scores)
                    avgs = torch.cat(all_avgs)
                else:
                    scores, avgs = batch_evaluate(result, raw_data, target_ret, device)
                
                # 收集结果
                op1_id = FORMULA_VOCAB.token_names.index(op1_name)
                op2_id = FORMULA_VOCAB.token_names.index(op2_name)
                op3_id = FORMULA_VOCAB.token_names.index(op3_name)
                
                for i in range(N):
                    if structure_id == 0:
                        formula = [all_combos[i, 0].item(), all_combos[i, 1].item(), op1_id,
                                   all_combos[i, 2].item(), all_combos[i, 3].item(), op3_id, op2_id]
                    elif structure_id == 1:
                        formula = [all_combos[i, 0].item(), all_combos[i, 1].item(), op1_id,
                                   all_combos[i, 2].item(), op2_id, all_combos[i, 3].item(), op3_id]
                    elif structure_id == 2:
                        formula = [all_combos[i, 0].item(), all_combos[i, 1].item(), all_combos[i, 2].item(),
                                   op2_id, op1_id, all_combos[i, 3].item(), op3_id]
                    elif structure_id == 3:
                        formula = [all_combos[i, 0].item(), all_combos[i, 1].item(), all_combos[i, 2].item(),
                                   op2_id, all_combos[i, 3].item(), op3_id, op1_id]
                    elif structure_id == 4:
                        formula = [all_combos[i, 0].item(), all_combos[i, 1].item(), all_combos[i, 2].item(),
                                   all_combos[i, 3].item(), op3_id, op2_id, op1_id]
                    
                    results.append({
                        'formula_ids': formula,
                        'score': scores[i].item(),
                        'avg_ret': avgs[i].item(),
                    })
                
                # 释放中间结果显存
                del t1, t2, result
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
    
    return results


def main():
    print("=" * 70)
    print("7-Token 穷举搜索（GPU 批量版，仅特征 token + 二元算子）")
    print(f"训练区间: {ModelConfig.TRAIN_START_DATE} ~ {ModelConfig.TRAIN_END_DATE or 'latest'}")
    print(f"指定股票: {ModelConfig.TRAIN_SPECIFIC_CODES}")
    print("=" * 70)
    
    # 检查 GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n设备: {device}")
    if device.type == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("  警告: CUDA 不可用，回退到 CPU（会很慢）")
    
    # 1. 加载数据
    print("\n[1/3] 加载数据...")
    loader = AShareDataLoader()
    loader.load_data(
        start_date=ModelConfig.TRAIN_START_DATE,
        end_date=ModelConfig.TRAIN_END_DATE,
        specific_codes=ModelConfig.TRAIN_SPECIFIC_CODES
    )
    print(f"  完成: {len(loader.codes)} 只股票, {len(loader.dates)} 个交易日")
    
    # 2. 数据移到 GPU
    print(f"\n[2/3] 数据移到 {device}...")
    feat_tensor = loader.feat_tensor.to(device)
    raw_data = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                for k, v in loader.raw_data_cache.items()}
    target_ret = loader.target_ret.to(device)
    
    # 3. 确定可用 token
    feature_ids = list(range(FORMULA_VOCAB.operator_offset))  # 0-11
    binary_op_names = [cfg[0] for cfg in OPS_CONFIG if cfg[2] == 2]
    
    print(f"\n  二元算子: {binary_op_names} ({len(binary_op_names)} 个)")
    print(f"  特征 token: {[FORMULA_VOCAB.token_names[i] for i in feature_ids]} ({len(feature_ids)} 个)")
    
    total_formulas = 5 * len(feature_ids)**4 * len(binary_op_names)**3
    print(f"\n  7-token 总组合数: {total_formulas:,}")
    print(f"  每种结构: {len(feature_ids)**4 * len(binary_op_names)**3:,} 种")
    
    # 4. GPU 批量穷举
    print(f"\n[3/3] GPU 批量穷举 5 种结构...")
    all_results = []
    start_time = time.time()
    
    for structure_id in range(5):
        print(f"\n  结构 {structure_id + 1}/5...")
        struct_results = brute_force_structure_gpu(
            structure_id, feat_tensor, raw_data, target_ret,
            feature_ids, binary_op_names, device
        )
        all_results.extend(struct_results)
        print(f"    完成: {len(struct_results)} 个结果")
    
    elapsed = time.time() - start_time
    print(f"\n  全部完成: {len(all_results)}/{total_formulas} 有效，耗时 {elapsed:.1f}s")
    
    # 5. 排序并输出
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
    
    # 6. 统计
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
    
    # 7. 保存结果
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'brute_force_7token_gpu_results.json')
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
    main()
