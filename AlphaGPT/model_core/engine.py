import torch
from torch.distributions import Categorical
from tqdm import tqdm
import json

from .config import ModelConfig
from .data_loader import AShareDataLoader
from .alphagpt import AlphaGPT, NewtonSchulzLowRankDecay, StableRankMonitor
from .vm import StackVM
from .backtest import AShareBacktest

class AShareAlphaEngine:
    """A 股 AlphaGPT 训练引擎"""
    
    def __init__(self, use_lord_regularization=True, lord_decay_rate=1e-3, lord_num_iterations=5):
        self.loader = AShareDataLoader()
        self.loader.load_data()
        
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
    
    def train(self):
        print(">>> Starting A-Share Alpha Mining with LoRD Regularization..." if self.use_lord else ">>> Starting A-Share Alpha Mining...")
        if self.use_lord:
            print(f"   LoRD Regularization enabled")
            print(f"   Target keywords: ['q_proj', 'k_proj', 'attention', 'qk_norm']")
        
        pbar = tqdm(range(ModelConfig.TRAIN_STEPS))
        
        for step in pbar:
            bs = ModelConfig.BATCH_SIZE
            inp = torch.zeros((bs, 1), dtype=torch.long, device=ModelConfig.DEVICE)
            
            log_probs = []
            tokens_list = []
            
            for _ in range(ModelConfig.MAX_FORMULA_LEN):
                logits, _, _ = self.model(inp)
                dist = Categorical(logits=logits)
                action = dist.sample()
                
                log_probs.append(dist.log_prob(action))
                tokens_list.append(action)
                inp = torch.cat([inp, action.unsqueeze(1)], dim=1)
            
            seqs = torch.stack(tokens_list, dim=1)
            
            rewards = torch.zeros(bs, device=ModelConfig.DEVICE)
            
            for i in range(bs):
                formula = seqs[i].tolist()
                
                # 尝试关键前缀长度（奇数长度更可能有效），减少搜索次数
                res = None
                for trunc_len in [1, 3, 5, 7, 9, 11, 13, 14]:
                    if trunc_len > len(formula):
                        continue
                    candidate = formula[:trunc_len]
                    res = self.vm.execute(candidate, self.loader.feat_tensor)
                    if res is not None:
                        break
                
                if res is None:
                    rewards[i] = -5.0
                    continue
                
                if res.std() < 1e-4:
                    rewards[i] = -2.0
                    continue
                
                score, ret_val = self.bt.evaluate(res, self.loader.raw_data_cache, self.loader.target_ret)
                rewards[i] = score
                
                if score.item() > self.best_score:
                    self.best_score = score.item()
                    self.best_formula = formula[:trunc_len]  # 保存实际有效的公式
                    tqdm.write(f"[!] New King: Score {score:.2f} | Ret {ret_val:.2%} | Formula {formula[:trunc_len]}")
            
            # 归一化奖励
            adv = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
            
            loss = 0
            for t in range(len(log_probs)):
                loss += -log_probs[t] * adv
            
            loss = loss.mean()
            
            # 梯度更新
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
            
            if self.use_lord:
                self.lord_opt.step()
            
            # 日志
            avg_reward = rewards.mean().item()
            postfix_dict = {'AvgRew': f"{avg_reward:.3f}", 'BestScore': f"{self.best_score:.3f}"}
            
            if self.use_lord and step % 100 == 0:
                stable_rank = self.rank_monitor.compute()
                postfix_dict['Rank'] = f"{stable_rank:.2f}"
                self.training_history['stable_rank'].append(stable_rank)
            
            self.training_history['step'].append(step)
            self.training_history['avg_reward'].append(avg_reward)
            self.training_history['best_score'].append(self.best_score)
            
            pbar.set_postfix(postfix_dict)
        
        # 保存最优公式
        with open("best_meme_strategy.json", "w") as f:
            json.dump(self.best_formula, f)
        
        with open("training_history.json", "w") as f:
            json.dump(self.training_history, f)
        
        print(f"\n[OK] Training completed!")
        print(f"  Best score: {self.best_score:.4f}")
        print(f"  Best formula: {self.best_formula}")


if __name__ == "__main__":
    eng = AShareAlphaEngine(use_lord_regularization=True)
    eng.train()
