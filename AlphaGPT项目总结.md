# AlphaGPT 项目深度总结

> 项目地址：https://github.com/imbue-bit/AlphaGPT  
> 文档生成时间：2026年7月13日

---

## 一、项目概述

**AlphaGPT** 是一套面向 **Solana Meme 币生态** 的端到端量化交易系统，核心设计理念是：**不直接预测价格，而是让 AI 自动挖掘可解释的交易因子公式**。

项目采用"生成公式 → 解释执行 → 回测评分 → 优化生成器"的闭环训练范式，将策略研究与交易执行清晰分层。据 README 所述，开源版本曾管理过 **1.66 亿人民币** 规模的资金。

---

## 二、系统架构总览

```
┌─────────────────────────────────────────────────────────────────┐
│                        AlphaGPT 系统架构                         │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────────────┐   │
│  │ data_pipeline│   │ model_core  │   │  strategy_manager   │   │
│  │   数据管线    │ → │  策略挖掘   │ → │    实盘执行引擎      │   │
│  └─────────────┘   └─────────────┘   └─────────────────────┘   │
│         ↓                  ↓                   ↓                  │
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────────────┐   │
│  │  Birdeye    │   │  AlphaGPT   │   │   SolanaTrader      │   │
│  │  DexScreener│   │  Transformer│   │   (Jupiter/RPC)     │   │
│  └─────────────┘   └─────────────┘   └─────────────────────┘   │
│         ↓                  ↓                   ↓                  │
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────────────┐   │
│  │ Postgres/    │   │ best_meme_  │   │   portfolio_state   │   │
│  │ TimescaleDB  │   │ strategy.json│   │      .json          │   │
│  └─────────────┘   └─────────────┘   └─────────────────────┘   │
│                                                                   │
│  ┌─────────────────────────────────────────────────────────┐     │
│  │              dashboard/  (Streamlit 看板)                  │     │
│  └─────────────────────────────────────────────────────────┘     │
└─────────────────────────────────────────────────────────────────┘
```

---

## 三、核心模块详解

### 3.1 model_core — 策略挖掘引擎

这是整个系统最精华的部分，负责**自动生成、评估、优化交易因子公式**。

#### 3.1.1 核心思想：公式即 Token 序列

AlphaGPT 将交易因子公式看作一种**领域特定语言（DSL）**，每个公式是由 "特征 token" 和 "算子 token" 组成的序列。模型通过自回归方式生成 token 序列，再用虚拟机执行得到因子信号。

**词汇表（Vocab）** 定义于 `vocab.py`：

| 类型 | 名称 | 说明 |
|------|------|------|
| 特征 | `RET`, `LIQ_SCORE`, `PRESSURE`, `FOMO`, `DEV`, `LOG_VOL` | 6 个基础因子（可扩展至 12 个） |
| 算子 | `ADD`, `SUB`, `MUL`, `DIV`, `NEG`, `ABS`, `SIGN`, `GATE`, `JUMP`, `DECAY`, `DELAY1`, `MAX3` | 12 个时序/数学算子 |

#### 3.1.2 模型架构：AlphaGPT（`alphagpt.py`）

```
Input: token sequence [B, T]
  ↓
Token Embedding + Positional Embedding
  ↓
Looped Transformer (2 layers, 4 heads, d_model=64, loop=3)
  - QKNorm: Query-Key 归一化，稳定注意力
  - RMSNorm: 替代 LayerNorm
  - SwiGLU: 替代标准 FFN
  ↓
RMSNorm
  ↓
┌──────────────┬─────────────┐
│  MTPHead     │  Critic Head │
│ 多任务池化头  │  价值评估头   │
│ (3 个任务)   │  (d_model→1) │
└──────────────┴─────────────┘
```

**关键创新点：**

1. **Looped Transformer Layer**：每个 Transformer 层内部循环 3 次，实现"循环精炼"（recurrent refinement），用更少层数获得更深表达能力。

2. **QKNorm（Query-Key Normalization）**：对 Attention 的 Q、K 做 L2 归一化，消除量级差异导致的注意力崩塌。

3. **LoRD（Low-Rank Decay）正则化**：使用 **Newton-Schulz 迭代**（5 次迭代）逼近最小奇异向量，对 attention 参数施加低秩衰减，无需显式 SVD。目标关键词：`q_proj`, `k_proj`, `attention`, `qk_norm`。

4. **MTPHead（Multi-Task Pooling Head）**：3 个任务头 + 任务路由器，通过门控机制加权组合输出，支持多目标学习。

5. **StableRankMonitor**：监控参数的有效秩（stable rank = ||W||_F² / ||W||_2²），追踪模型复杂度。

#### 3.1.3 因子体系（`factors.py`）

**基础因子（6 维）—— `FeatureEngineer`：**

| 因子 | 计算方式 | 含义 |
|------|---------|------|
| `ret` | `log(close_t / close_{t-1})` | 对数收益率 |
| `liq_score` | `clamp(liquidity/fdv * 4, 0, 1)` | 流动性健康度 |
| `pressure` | `tanh((close-open)/(high-low) * 3)` | 买卖力量不平衡（K线实体/振幅） |
| `fomo` | `volume 加速度的差分` | 成交量加速度（FOMO 情绪） |
| `dev` | `(close - MA20) / MA20` | 价格偏离均值程度 |
| `log_vol` | `log(1 + volume)` | 对数成交量 |

**扩展因子（12 维）—— `AdvancedFactorEngineer`：**

在基础因子基础上增加：
- `vol_cluster`: 波动率聚集（GARCH 思想）
- `momentum_rev`: 动量反转检测
- `rel_strength`: RSI 类相对强弱指标
- `hl_range`: 高低价振幅
- `close_pos`: 收盘价在区间中的位置
- `vol_trend`: 成交量趋势

**鲁棒归一化**：使用 `Median + MAD`（中位数绝对偏差）替代 Z-Score，对极端值更鲁棒。

#### 3.1.4 算子系统（`ops.py`）

所有算子均用 `@torch.jit.script` 编译，提升执行效率：

| 算子 | 元数 | 功能 | 公式 |
|------|------|------|------|
| `ADD` | 2 | 加法 | `x + y` |
| `SUB` | 2 | 减法 | `x - y` |
| `MUL` | 2 | 乘法 | `x * y` |
| `DIV` | 2 | 安全除法 | `x / (y + 1e-6)` |
| `NEG` | 1 | 取负 | `-x` |
| `ABS` | 1 | 绝对值 | `abs(x)` |
| `SIGN` | 1 | 符号 | `sign(x)` |
| `GATE` | 3 | 门控选择 | `condition>0 ? x : y` |
| `JUMP` | 1 | 极端跳变检测 | `relu(zscore - 3)` |
| `DECAY` | 1 | 衰减叠加 | `x + 0.8*x_{t-1} + 0.6*x_{t-2}` |
| `DELAY1` | 1 | 滞后一期 | `x_{t-1}` |
| `MAX3` | 1 | 三期最大 | `max(x_t, x_{t-1}, x_{t-2})` |

#### 3.1.5 公式虚拟机（`vm.py`）

`StackVM` 采用**栈式执行模型**：

1. 遍历 token 序列
2. 遇到特征 token：将对应特征矩阵压栈
3. 遇到算子 token：弹出指定数量的操作数，执行运算后将结果压栈
4. 最终栈中只剩一个结果：即为该公式生成的因子信号
5. 异常处理（栈下溢、NaN/Inf、非法 token）返回 `None` 作为惩罚信号

```python
# 示例公式（token 序列）
# [RET, LIQ_SCORE, MUL, PRESSURE, ADD] 
# 含义: (ret * liq_score) + pressure
```

#### 3.1.6 回测引擎（`backtest.py`）

`MemeBacktest` 设计针对 Meme 币特点：

- **交易规模**: 1000 USD/笔
- **最小流动性**: 500,000 USD（低于视为无法交易）
- **基础费率**: 0.6%（Swap + Gas + Jito Tip）
- **冲击滑点**: `trade_size / liquidity`，上限 5%
- **信号转换**: `sigmoid(factors)` 将因子转为 0~1 概率
- **开平仓**: `signal > 0.85` 且流动性安全时开仓
- **换手率惩罚**: 根据 turnover 计算交易成本
- **大回撤惩罚**: 单笔亏损 > 5% 时额外扣分
- **活跃度过滤**: 交易次数 < 5 的公式得分 = -10

**适应度函数**:
```
score = cum_ret - (big_drawdowns * 2.0)
final_fitness = median(score)  # 用中位数而非均值，更鲁棒
```

#### 3.1.7 训练引擎（`engine.py`）

采用 **REINFORCE 策略梯度** 算法：

```python
for step in range(1000):
    # 1. 生成公式 (BATCH_SIZE=8192 个并行)
    seqs = model.generate_batch()
    
    # 2. VM 执行 + 回测评分
    rewards = [backtest(vm.execute(formula)) for formula in seqs]
    
    # 3. 奖励归一化（优势函数）
    adv = (rewards - mean) / (std + 1e-5)
    
    # 4. 策略梯度损失
    loss = -sum(log_prob * adv)
    
    # 5. 标准梯度更新 + LoRD 正则化
    optimizer.step()
    lord_opt.step()
```

**惩罚机制**（针对无效公式）：
- VM 执行失败 → 奖励 = -5.0
- 因子标准差 < 1e-4（无区分度）→ 奖励 = -2.0

---

### 3.2 data_pipeline — 数据管线

#### 3.2.1 数据来源

| 数据源 | 用途 | 说明 |
|--------|------|------|
| **Birdeye** | 主要数据源 |  trending tokens + OHLCV，需 API Key |
| **DexScreener** | 辅助/备用 | 可选启用 |

#### 3.2.2 数据筛选逻辑（`data_manager.py`）

```python
筛选条件：
- liquidity >= 500,000 USD      # 最小流动性
- fdv >= 10,000,000 USD         # 最小市值
- fdv <= MAX_FDV (默认 inf)     # 剔除 WIF/BONK 等巨无霸，专注早期高成长
```

#### 3.2.3 数据库设计（`db_manager.py`）

**Postgres/TimescaleDB** 双模式：

```sql
-- tokens 表：代币元信息
CREATE TABLE tokens (
    address TEXT PRIMARY KEY,
    symbol TEXT, name TEXT, decimals INT,
    chain TEXT, last_updated TIMESTAMP
);

-- ohlcv 表：时序行情（支持 TimescaleDB Hypertable）
CREATE TABLE ohlcv (
    time TIMESTAMP NOT NULL,
    address TEXT NOT NULL,
    open, high, low, close DOUBLE PRECISION,
    volume, liquidity, fdv DOUBLE PRECISION,
    source TEXT,
    PRIMARY KEY (time, address)
);
```

#### 3.2.4 数据加载（`data_loader.py`）

从 SQL 读取数据后：
1. 按 `address` pivot 成 `[Tokens, Time]` 矩阵
2. 前向填充缺失值，空缺填 0
3. 转为 PyTorch Tensor（GPU）
4. 计算特征张量 `feat_tensor: [Tokens, Features, Time]`
5. 目标收益：`target_ret = log(open_{t+2} / open_{t+1})`（用未来第2期开盘/未来第1期开盘）

---

### 3.3 strategy_manager — 实盘策略执行

#### 3.3.1 主循环（`runner.py`）

```python
while True:
    # 1. 每 15 分钟同步数据管线
    if time_since_last_sync > 900s:
        await data_mgr.pipeline_sync_daily()
    
    # 2. 加载最新数据（Top 300 代币）
    loader.load_data(limit_tokens=300)
    
    # 3. 监控持仓（止损/止盈/移动止损/AI 退出）
    await monitor_positions()
    
    # 4. 扫描入场机会（如果仓位未满）
    if open_positions < MAX_OPEN_POSITIONS:
        await scan_for_entries()
    
    # 5. 睡眠 60 秒
    sleep(60)
```

#### 3.3.2 入场逻辑（`scan_for_entries`）

1. 使用最优公式对 300 个代币生成最新信号分数
2. `sigmoid(raw_signal)` 转为 0~1 概率
3. 按分数从高到低排序
4. 分数 ≥ `BUY_THRESHOLD` (0.85) 的代币进入候选
5. **RiskEngine 安全检查**：
   - 流动性 ≥ 5000 USD
   - 通过 Jupiter 模拟卖出路径（防 honeypot）
6. 通过检查的代币执行买入

#### 3.3.3 出场逻辑（`monitor_positions`）

四重退出机制：

| 触发条件 | 操作 | 说明 |
|----------|------|------|
| **止损** | 清仓 | `pnl <= -5%` |
| **止盈目标1** | 卖 50% | `pnl >= +10%`，标记为 moonbag（让利润奔跑） |
| **移动止损** | 清仓 | 最大盈利 > 5% 后回撤 > 3% |
| **AI 信号** | 清仓 | 当前 AI 分数 < `SELL_THRESHOLD` (0.45) |

#### 3.3.4 风控引擎（`risk.py`）

- **流动性检查**：`liquidity >= 5000 USD`
- **Honeypot 检测**：通过 Jupiter 询价验证卖出路径是否通畅
- **仓位管理**：固定每笔投入 2 SOL（或按余额动态调整）
- **最大持仓**：3 个并发仓位

#### 3.3.5 持仓管理（`portfolio.py`）

基于 `dataclass` 的持仓对象，持久化到 `portfolio_state.json`：

```python
@dataclass
class Position:
    token_address: str
    symbol: str
    entry_price: float        # 入场价格（SOL计价）
    entry_time: float         # 入场时间戳
    amount_held: float        # 持仓数量
    initial_cost_sol: float   # 初始投入 SOL
    highest_price: float      # 历史最高价（用于回撤计算）
    is_moonbag: bool = False  # 是否已翻倍出本
```

---

### 3.4 execution — 交易执行层

#### 3.4.1 架构

```
StrategyRunner
    └── SolanaTrader
            ├── QuickNodeClient  (Solana RPC)
            │       └── get_balance, send_and_confirm
            └── JupiterAggregator
                    ├── get_quote   (询价)
                    ├── get_swap_tx (获取交易体)
                    └── deserialize_and_sign (签名)
```

#### 3.4.2 关键流程

**买入流程：**
1. 检查 SOL 余额是否充足
2. Jupiter 询价：`SOL → Token`（输入 lamports，获取预估输出）
3. 获取 swap 交易体（base64）
4. 反序列化 + 私钥签名
5. RPC 发送并确认交易
6. 用 quote 的预估输出记账（避免链上余额查询延迟）

**卖出流程：**
1. 查询 token 账户余额（`get_token_accounts_by_owner_json_parsed`）
2. Jupiter 询价：`Token → SOL`
3. 获取 swap 交易体并签名发送
4. 更新/关闭持仓

#### 3.4.3 配置要点（`execution/config.py`）

- `RPC_URL`: QuickNode 或公共 RPC
- `SOLANA_PRIVATE_KEY`: 钱包私钥（Base58 或 JSON 数组格式）
- `DEFAULT_SLIPPAGE_BPS`: 默认滑点 2%（200 bps）
- `SOL_MINT`: SOL 代币地址
- `USDC_MINT`: USDC 代币地址

---

### 3.5 dashboard — Streamlit 监控看板

基于 **Streamlit** 的实时监控界面，功能包括：

1. **钱包状态**：实时 SOL 余额
2. **紧急停止**：一键写入 `STOP_SIGNAL` 文件终止交易循环
3. **持仓面板**：
   - 当前持仓数量 / 最大 5 个
   - 总投入 SOL
   - 未实现盈亏（基于 highest_price 估算）
   - 各仓位 PnL 柱状图
4. **市场扫描器**：流动性 vs 成交量散点图（气泡大小 = FDV）
5. **系统日志**：最近 20 条日志实时展示
6. **自动刷新**：30 秒周期自动刷新

---

## 四、技术依赖

| 类别 | 库 | 版本 | 用途 |
|------|-----|------|------|
| 深度学习 | `torch` | ≥2.0.0 | 模型训练与推理 |
| 数值计算 | `numpy` | ≥1.24.0 | 数值运算 |
| 数据处理 | `pandas` | ≥2.0.0 | 数据操作 |
| 数据库 | `sqlalchemy`, `asyncpg`, `psycopg2-binary` | ≥2.0.0 | Postgres 交互 |
| 异步 HTTP | `aiohttp` | ≥3.9.0 | API 请求 |
| Solana | `solders`, `solana`, `base58` | ≥0.18/0.30 | 区块链交互 |
| 看板 | `streamlit`, `plotly` | ≥1.28/5.17 | 可视化界面 |
| 日志 | `loguru`, `tqdm` | ≥0.7/4.66 | 日志与进度 |
| 环境配置 | `python-dotenv` | ≥1.0.0 | .env 管理 |

---

## 五、运行流程

### 5.1 训练阶段（离线）

```bash
# 1. 准备数据：确保 Postgres 中有 ohlcv 数据
python -m data_pipeline.run_pipeline

# 2. 启动训练
python -m model_core.engine
# 输出: best_meme_strategy.json, training_history.json
```

### 5.2 实盘阶段（在线）

```bash
# 1. 配置 .env 文件（DB、Birdeye API、Solana 私钥、RPC）

# 2. 启动数据管线（后台或定时）
python -m data_pipeline.run_pipeline

# 3. 启动策略执行器
python -m strategy_manager.runner

# 4. （可选）启动监控看板
streamlit run dashboard/app.py
```

---

## 六、项目亮点与可借鉴之处

### 6.1 架构设计

1. **清晰的分层架构**：数据层、模型层、策略层、执行层完全解耦，每层可独立运行和测试。
2. **"公式生成"范式**：不黑盒预测价格，而是生成可解释的因子公式，兼具可解释性和灵活性。
3. **StackVM 执行模型**：轻量、高效、可扩展，新增算子只需在 `ops.py` 中注册即可。

### 6.2 模型创新

1. **Looped Transformer**：层内循环减少参数量，同时增加等效深度。
2. **LoRD 正则化**：Newton-Schulz 迭代实现低秩衰减，无需 SVD，计算高效。
3. **QKNorm + RMSNorm**：稳定 Attention 训练，缓解 Transformer 的收敛问题。
4. **MTPHead**：多任务学习，支持同时优化多个目标（如收益、夏普、回撤）。

### 6.3 工程细节

1. **鲁棒归一化**：Median + MAD 替代 Z-Score，对 Meme 币的极端价格更鲁棒。
2. **中位数适应度**：回测评分用 median 而非 mean，降低异常样本影响。
3. **Honeypot 检测**：通过 Jupiter 模拟卖出路径，避免买入无法卖出的代币。
4. **Moonbag 策略**：翻倍出本后让利润奔跑，典型的 Meme 币止盈策略。
5. **STOP_SIGNAL 机制**：文件信号实现安全停机，适配异步事件循环。

### 6.4 风控设计

1. **流动性过滤**：多层流动性检查（数据筛选 500K、回测 500K、交易前 5K）。
2. **滑点建模**：基础费率 + 冲击滑点，更贴近真实交易成本。
3. **四重出场机制**：止损 + 止盈 + 移动止损 + AI 信号，全方位保护。
4. **仓位上限**：最多 3 个并发持仓，控制集中度风险。

---

## 七、潜在局限与注意事项

1. **依赖外部服务**：Birdeye API（付费）、Postgres、Solana RPC、Jupiter，任一故障都会影响系统。
2. **缺少完整 .env 模板**：实盘运行需要自行配置多个密钥和地址。
3. **训练需要数据**：`best_meme_strategy.json` 需先训练生成，仓库默认不带预训练模型。
4. **面向 Solana 生态**：代码深度耦合 Jupiter/Solana，难以直接迁移到以太坊或 A 股。
5. **Meme 币高风险**：策略设计针对高波动 Meme 币，不适用于低波动标的。
6. **作者曾受安全威胁**：项目 README 提到作者受到人身安全威胁，使用时需注意风险。

---

## 八、关键文件索引

| 文件 | 职责 | 核心类/函数 |
|------|------|------------|
| `model_core/alphagpt.py` | 模型定义 | `AlphaGPT`, `LoopedTransformer`, `LoRD`, `MTPHead` |
| `model_core/engine.py` | 训练引擎 | `AlphaEngine.train()` |
| `model_core/factors.py` | 特征工程 | `FeatureEngineer`, `AdvancedFactorEngineer`, `MemeIndicators` |
| `model_core/vm.py` | 公式虚拟机 | `StackVM.execute()` |
| `model_core/backtest.py` | 回测评估 | `MemeBacktest.evaluate()` |
| `model_core/ops.py` | 算子定义 | `OPS_CONFIG` |
| `data_pipeline/data_manager.py` | 数据同步 | `DataManager.pipeline_sync_daily()` |
| `data_pipeline/db_manager.py` | 数据库管理 | `DBManager` |
| `strategy_manager/runner.py` | 实盘主循环 | `StrategyRunner.run_loop()` |
| `strategy_manager/risk.py` | 风控引擎 | `RiskEngine.check_safety()` |
| `strategy_manager/portfolio.py` | 持仓管理 | `PortfolioManager` |
| `execution/trader.py` | 交易封装 | `SolanaTrader.buy/sell()` |
| `execution/jupiter.py` | Jupiter 交互 | `JupiterAggregator` |
| `execution/rpc_handler.py` | RPC 客户端 | `QuickNodeClient` |
| `dashboard/app.py` | 监控看板 | Streamlit UI |

---

## 九、总结

AlphaGPT 是一个**工程完成度较高**的加密量化交易系统，其最大价值在于**"自动因子挖掘"**的范式创新：

- 用 **Transformer 生成公式** 替代人工设计因子
- 用 **StackVM 执行** 保证可解释性和灵活性
- 用 **回测奖励** 驱动模型自优化
- 用 **清晰分层** 实现研究到生产的无缝衔接

对于量化研究者而言，最值得借鉴的是其**"公式生成 + 虚拟机执行 + 回测评分"**的闭环设计，以及**Looped Transformer + LoRD 正则化**在时序因子挖掘中的应用。对于实盘开发者，其**风控分层、Honeypot 检测、Moonbag 策略**等工程细节也颇具参考价值。

---

*文档基于 AlphaGPT 仓库代码逐行阅读整理，覆盖全部核心模块。*
