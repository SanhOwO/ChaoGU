import torch
from torch.distributions import Categorical
from tqdm import tqdm
import json
import os

from .vocab import FORMULA_VOCAB
from .config import ModelConfig
from .data_loader import AShareDataLoader
from .alphagpt import AlphaGPT, NewtonSchulzLowRankDecay, StableRankMonitor
from .vm import StackVM
from .backtest import AShareBacktest

# 输出目录：项目根目录下的 个股训练数据/
# engine.py 位于 AlphaGPT/model_core/，向上两级到项目根目录
OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "个股训练数据"
)
os.makedirs(OUTPUT_DIR, exist_ok=True)

class AShareAlphaEngine:
    """A 股 AlphaGPT 训练引擎"""
    
    def __init__(self, use_lord_regularization=True, lord_decay_rate=1e-3, lord_num_iterations=5):
        self.loader = AShareDataLoader()
        self.loader.load_data(
            start_date=ModelConfig.TRAIN_START_DATE,
            end_date=ModelConfig.TRAIN_END_DATE,
            specific_codes=getattr(ModelConfig, 'TRAIN_SPECIFIC_CODES', None)
        )
        
        self.model = AlphaGPT().to(ModelConfig.DEVICE)
        
        # 标准优化器
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=1e-3)
        
        # LoRD 正则化
        self.use_lord = use_lord_regularization
        if self.use_lord:
            self.lord_opt = NewtonSchulzLowRankDecay(
                self.model.named_parameters(),
                decay_rate=lord_decay_rate,
                num_iterations=lord_num_iterations,
                target_keywords=["q_proj", "k_proj", "attention", "qk_norm"]
            )
            self.rank_monitor = StableRankMonitor(
                self.model,
                target_keywords=["q_proj", "k_proj"]
            )
        else:
            self.lord_opt = None
            self.rank_monitor = None
        
        self.vm = StackVM()
        self.bt = AShareBacktest()
        
        self.best_score = -float('inf')
        self.best_formula = None
        self.training_history = {
            'step': [],
            'avg_reward': [],
            'best_score': [],
            'stable_rank': []
        }
    
    def train(self, save_prefix=None):
        """
        训练 AlphaGPT
        
        Args:
            save_prefix: 保存文件名前缀（如 '300633'），
                        为 None 时使用默认文件名
        """
        # 动态生成保存文件名
        if save_prefix:
            strategy_file = os.path.join(OUTPUT_DIR, f"{save_prefix}_strategy.json")
            history_file = os.path.join(OUTPUT_DIR, f"{save_prefix}_training_history.json")
        else:
            strategy_file = os.path.join(OUTPUT_DIR, "best_meme_strategy.json")
            history_file = os.path.join(OUTPUT_DIR, "training_history.json")
        
        print(f">>> Starting A-Share Alpha Mining with LoRD Regularization..." if self.use_lord else ">>> Starting A-Share Alpha Mining...")
        print(f"   Strategy will be saved to: {strategy_file}")
        print(f"   History will be saved to: {history_file}")
        if self.use_lord:
            print(f"   LoRD Regularization enabled")
            print(f"   Target keywords: ['q_proj', 'k_proj', 'attention', 'qk_norm']")
        
        pbar = tqdm(range(ModelConfig.TRAIN_STEPS))
        
        for step in pbar:
            bs = ModelConfig.BATCH_SIZE
            inp = torch.zeros((bs, 1), dtype=torch.long, device=ModelConfig.DEVICE)
            
            log_probs = []
            tokens_list = []
            entropy_list = []  # 收集每步的熵，用于防止过早收敛
            
            for _ in range(ModelConfig.MAX_FORMULA_LEN):
                logits, _, _ = self.model(inp)
                dist = Categorical(logits=logits)
                action = dist.sample()
                
                log_probs.append(dist.log_prob(action))
                tokens_list.append(action)
                entropy_list.append(dist.entropy())  # 收集熵，保持探索
                inp = torch.cat([inp, action.unsqueeze(1)], dim=1)
            
            seqs = torch.stack(tokens_list, dim=1)
            
            rewards = torch.zeros(bs, device=ModelConfig.DEVICE)
            raw_scores = [-5.0] * bs  # 默认值：无效公式
            formula_details = []  # 收集每步所有公式详情，用于打印 top 3
            
            for i in range(bs):
                formula = seqs[i].tolist()
                
                # 尝试关键前缀长度（Polish notation 只可能奇数长度有效），减少搜索次数
                # 方案A：强制至少5个token，去掉trunc_len=1和3，防止模型被困在短公式
                res = None
                for trunc_len in [5, 7, 9, 11, 13]:
                    if trunc_len > len(formula):
                        continue
                    candidate = formula[:trunc_len]
                    res = self.vm.execute(candidate, self.loader.feat_tensor)
                    if res is not None:
                        break
                
                if res is None:
                    continue  # raw_scores[i] 保持 -5.0（无效公式）
                
                # 注：原项目此处有 res.std() < 1e-4 检查，过滤"无区分度"公式
                # 但加了常数 token 后，CONST_NEG10（std=0）是合法的"不交易"策略，
                # 若继续过滤，模型永远发现不了 score=0 的最优基线，故删除
                
                score, ret_val = self.bt.evaluate(res, self.loader.raw_data_cache, self.loader.target_ret)
                
                # 方案A：强制至少5-token后，无需再惩罚单token常数策略
                
                raw_scores[i] = score.item()
                
                # 收集公式详情（用于打印 top 3）
                formula_details.append({
                    'formula': candidate,
                    'score': score.item(),
                    'ret': ret_val,
                    'is_constant': len(candidate) == 1 and FORMULA_VOCAB.get_constant_value(candidate[0]) is not None
                })
                
                if score.item() > self.best_score:
                    self.best_score = score.item()
                    self.best_formula = formula[:trunc_len]  # 保存实际有效的公式
                    tqdm.write(f"[!] New King: Score {score:.2f} | Ret {ret_val:.2%} | Formula {formula[:trunc_len]}")
            
            # 动态门槛：随 best_score 提升，最低 0.064
            dynamic_threshold = max(0.064, self.best_score * 0.5)
            
            for i in range(bs):
                score_val = raw_scores[i]
                
                if score_val >= self.best_score:
                    rewards[i] = 1.0  # 新最优
                elif score_val > dynamic_threshold:
                    rewards[i] = 0.5  # 真正超越动态门槛
                else:
                    rewards[i] = -2.0  # 未达门槛（包括包装器）
            
            # 打印每步 top 3 公式详情（每 10 步打印一次）
            if formula_details and step % 10 == 0:
                formula_details.sort(key=lambda x: x['score'], reverse=True)
                stock_info = f"Stocks: {len(self.loader.codes)}" if len(self.loader.codes) > 1 else f"Stock: {self.loader.codes[0]}"
                tqdm.write(f"\n  Step {step} | {stock_info} | Top 3 Formulas:")
                for rank, fd in enumerate(formula_details[:3], 1):
                    const_tag = " [CONST]" if fd['is_constant'] else ""
                    tqdm.write(f"    #{rank} | Score: {fd['score']:+.4f} | Ret: {fd['ret']:+.2%} | Formula: {fd['formula']}{const_tag}")
            if formula_details and len(self.loader.codes) == 1 and step % 10 == 0:
                formula_details.sort(key=lambda x: x['score'], reverse=True)
                tqdm.write(f"\n  Step {step} | Stock: {self.loader.codes[0]} | Top 3 Formulas:")
                for rank, fd in enumerate(formula_details[:3], 1):
                    const_tag = " [CONST]" if fd['is_constant'] else ""
                    tqdm.write(f"    #{rank} | Score: {fd['score']:+.4f} | Ret: {fd['ret']:+.2%} | Formula: {fd['formula']}{const_tag}")
            
            # 归一化奖励
            adv = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
            
            # Policy Gradient Loss + Entropy Bonus（防止过早收敛到不交易）
            policy_loss = 0
            for t in range(len(log_probs)):
                policy_loss += -log_probs[t] * adv
            policy_loss = policy_loss.mean()
            
            # 熵正则化：鼓励模型保持 token 分布的多样性
            entropy = torch.stack(entropy_list, dim=1).mean()  # [bs, seq_len] -> scalar
            entropy_loss = -ModelConfig.ENTROPY_COEF * entropy
            
            loss = policy_loss + entropy_loss
            
            # 梯度更新
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
            
            if self.use_lord:
                self.lord_opt.step()
            
            # 日志
            avg_reward = rewards.mean().item()
            entropy_val = entropy.item()
            postfix_dict = {
                'AvgRew': f"{avg_reward:.3f}",
                'BestScore': f"{self.best_score:.3f}",
                'Threshold': f"{dynamic_threshold:.3f}",
                'Entropy': f"{entropy_val:.2f}"
            }
            
            if self.use_lord and step % 100 == 0:
                stable_rank = self.rank_monitor.compute()
                postfix_dict['Rank'] = f"{stable_rank:.2f}"
                self.training_history['stable_rank'].append(stable_rank)
            
            self.training_history['step'].append(step)
            self.training_history['avg_reward'].append(avg_reward)
            self.training_history['best_score'].append(self.best_score)
            
            # 每10步自动保存最优公式和训练历史，防止中断丢失
            if step % 10 == 0 and step > 0:
                with open(strategy_file, "w") as f:
                    json.dump(self.best_formula, f)
                with open(history_file, "w") as f:
                    json.dump(self.training_history, f)
            
            pbar.set_postfix(postfix_dict)
        
        # 保存最优公式
        with open(strategy_file, "w") as f:
            json.dump(self.best_formula, f)
        
        with open(history_file, "w") as f:
            json.dump(self.training_history, f)
        
        print(f"\n[OK] Training completed!")
        print(f"  Best score: {self.best_score:.4f}")
        print(f"  Best formula: {self.best_formula}")
        print(f"  Saved to: {strategy_file} & {history_file}")


if __name__ == "__main__":
    eng = AShareAlphaEngine(use_lord_regularization=True)
    eng.train()
