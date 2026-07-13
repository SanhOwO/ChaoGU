# AlphaGPT-AShare 使用说明

> **AlphaGPT A 股适配版** —— 基于 AlphaGPT 核心引擎的 A 股量化因子挖掘系统  
> 交易模式：**T 日收盘后 AI 分析 → T+1 日人工操作**  
> 数据源：**AKShare**（无需 API Key）

---

## 一、项目结构

```
AlphaGPT/
├── model_core/              # 模型核心（AlphaGPT 引擎）
│   ├── alphagpt.py          # Transformer 模型（Looped + LoRD + QKNorm）
│   ├── engine.py            # REINFORCE 训练引擎（A 股适配）
│   ├── factors.py           # A 股 12 维因子计算
│   ├── backtest.py          # A 股 T+1 回测引擎
│   ├── data_loader.py       # 从 SQLite 加载数据
│   ├── vm.py                # StackVM 公式虚拟机
│   ├── ops.py               # 12 个算子定义
│   ├── vocab.py             # A 股因子词汇表
│   └── config.py            # 模型配置
├── data_pipeline/           # 数据管线
│   ├── config.py            # A 股数据配置
│   ├── data_manager.py      # AKShare → SQLite 同步
│   └── providers/
│       └── akshare_provider.py  # AKShare 数据获取
├── strategy_manager/        # 策略执行（人工版）
│   ├── config.py            # 策略参数
│   └── signal_generator.py  # 收盘信号生成器
├── daily_runner.py          # 每日一键运行入口
├── requirements-ashare.txt  # A 股版本依赖
└── best_meme_strategy.json  # 训练产出的最优公式
```

---

## 二、安装

### 2.1 环境准备

```bash
# Python 3.10+ 推荐
cd AlphaGPT

# 安装依赖
pip install -r requirements-ashare.txt
```

### 2.2 依赖说明

| 库 | 用途 | 是否需要 API Key |
|----|------|-----------------|
| `torch` | 模型训练/推理 | ❌ |
| `akshare` | A 股行情数据 | ❌（完全免费） |
| `pandas` | 数据处理 | ❌ |
| `loguru` | 日志 | ❌ |
| `tqdm` | 进度条 | ❌ |

**整个系统没有任何需要付费 API Key 的组件。**

---

## 三、首次使用流程

### 3.1 第一步：下载历史数据

```bash
python daily_runner.py --mode sync
```

- 从 AKShare 下载全市场 A 股日 K 数据（前复权）
- 过滤条件：股价 2~500 元、流通市值 ≥ 1 亿、排除 ST
- 首次下载约 3 年历史数据，耗时 30~60 分钟
- 数据存储在 `data_pipeline/ashare_data.db`（SQLite）

### 3.2 第二步：训练模型

```bash
python daily_runner.py --mode train
```

- 使用 A 股历史数据训练 AlphaGPT
- 模型自动生成、评估、优化因子公式
- 训练产出：`best_meme_strategy.json`（最优公式）
- 默认 2000 步，约 10~30 分钟（取决于 GPU）

### 3.3 第三步：生成每日信号

```bash
python daily_runner.py --mode signal
```

- 使用最优公式对全市场股票打分
- 输出 `daily_report_YYYY-MM-DD.txt`（人工参考报告）
- 输出 `daily_report_YYYY-MM-DD_detail.csv`（完整数据）

---

## 四、日常使用流程（T 日收盘后）

### 4.1 自动模式（推荐）

```bash
# 一键执行：同步数据 → 生成信号
python daily_runner.py --mode full
```

### 4.2 分步模式

```bash
# Step 1: 同步当日数据（15:30 后运行）
python daily_runner.py --mode sync

# Step 2: 生成信号
python daily_runner.py --mode signal
```

### 4.3 查看报告

报告文件：`daily_report_YYYY-MM-DD.txt`

```
============================================================
  AlphaGPT-AShare 每日信号报告
  分析日期: 2025-07-13 (T 日)
  操作日期: T+1 日开盘
============================================================

📈 【重点关注】Top 买入信号（分数 >= 0.80）
------------------------------------------------------------

  Rank 1: 000001
    信号分数: 0.9234  🔥 强烈关注
    最新收盘价: 12.50 元
    流通市值: 1500.23 亿
    20日动量: 0.234  14日RSI: 62.5
    偏离均线: 0.056  量比: 1.85
    MACD柱: 0.012  波动率: 0.023

  Rank 2: 600519
    信号分数: 0.8912  🔥 强烈关注
    ...

📉 【回避信号】分数 <= 0.35
  300001: 分数 0.1234  ⚠️ 建议回避

📊 【市场统计】
  全市场分析股票数: 4823
  平均信号分数: 0.5123
  中位信号分数: 0.5087
  最高分: 0.9234 (000001)
  最低分: 0.0234 (600002)

============================================================
⚠️ 免责声明：本报告由 AI 模型生成，仅供研究参考，不构成投资建议。
   操作风险自负，请结合自身判断决策。
============================================================
```

---

## 五、A 股因子体系

### 5.1 12 维因子

| # | 因子 | 含义 | 计算方式 |
|---|------|------|----------|
| 1 | `RET` | 对数收益率 | `log(close_t / close_{t-1})` |
| 2 | `TURN` | 换手率标准化 | `换手率 / MA20换手率` |
| 3 | `VOLAT` | 20日波动率 | `std(RET, 20)` |
| 4 | `MOM` | 20日动量 | `sum(RET, 20)` |
| 5 | `LOG_AMT` | 对数成交额 | `log(1 + 成交额)` |
| 6 | `RSI` | 相对强弱 | `RSI(14)` |
| 7 | `DEV` | 偏离均线 | `(close - MA20) / MA20` |
| 8 | `VR` | 量比 | `成交量 / MA5成交量` |
| 9 | `LOG_MC` | 对数流通市值 | `log(1 + 流通市值)` |
| 10 | `HL_RANGE` | 振幅 | `(high - low) / close` |
| 11 | `CLOSE_POS` | 收盘位置 | `(close - low) / (high - low)` |
| 12 | `MACD_HIST` | MACD 柱状图 | `EMA12 - EMA26` 标准化 |

### 5.2 12 个算子

| 算子 | 元数 | 功能 |
|------|------|------|
| `ADD` | 2 | 加法 |
| `SUB` | 2 | 减法 |
| `MUL` | 2 | 乘法 |
| `DIV` | 2 | 安全除法 |
| `NEG` | 1 | 取负 |
| `ABS` | 1 | 绝对值 |
| `SIGN` | 1 | 符号 |
| `GATE` | 3 | 条件选择（condition>0 ? x : y）|
| `JUMP` | 1 | 极端跳变检测（zscore > 3）|
| `DECAY` | 1 | 衰减叠加（t + 0.8*t-1 + 0.6*t-2）|
| `DELAY1` | 1 | 滞后一期 |
| `MAX3` | 1 | 三期最大 |

---

## 六、训练参数调优

编辑 `model_core/config.py`：

```python
class ModelConfig:
    BATCH_SIZE = 4096          # 每批生成公式数（A 股约 5000 只股票，4096 合适）
    TRAIN_STEPS = 2000         # 训练步数（越多公式质量越高，但耗时更长）
    MAX_FORMULA_LEN = 14       # 公式最大长度（越长表达能力越强）
    
    # A 股交易成本
    BASE_FEE_BUY = 0.00025     # 买入佣金 0.025%
    BASE_FEE_SELL = 0.00075    # 卖出佣金+印花税 0.075%
    SLIPPAGE = 0.001           # 滑点 0.1%
```

---

## 七、策略参数调优

编辑 `strategy_manager/config.py`：

```python
class AShareStrategyConfig:
    BUY_THRESHOLD = 0.80       # 买入信号阈值（越高越严格）
    SELL_THRESHOLD = 0.35      # 卖出/回避阈值
    TOP_N_STOCKS = 20          # 每日关注股票数
    MAX_OPEN_POSITIONS = 5     # 最大持仓数（人工参考）
    STOP_LOSS_PCT = -0.05      # 止损线 -5%
    TAKE_PROFIT_PCT = 0.08     # 止盈线 +8%
```

---

## 八、与原版 AlphaGPT 的差异

| 维度 | 原版（Meme 币） | A 股适配版 |
|------|----------------|-----------|
| **数据源** | Birdeye API（付费）| AKShare（免费）|
| **数据库** | Postgres/TimescaleDB | SQLite（免部署）|
| **因子** | 6 维 Meme 特征 | 12 维 A 股特征 |
| **回测** | DEX 规则（即时成交）| T+1 规则（次日开盘成交）|
| **交易** | 自动链上执行 | **人工操作** |
| **目标收益** | open_{t+2} / open_{t+1} | 同上（A 股也适用）|
| **风控** | 流动性 + Honeypot | 涨跌停 + 成交额过滤 |
| **API Key** | Birdeye + QuickNode | **无需任何 Key** |

---

## 九、常见问题

### Q1: AKShare 数据获取失败？

AKShare 依赖东方财富等免费数据源，偶尔会因接口调整而失效。解决：
```bash
pip install --upgrade akshare
```

### Q2: 训练很慢？

- 有 GPU 则自动使用 CUDA，速度提升 5~10 倍
- 无 GPU 时减小 `BATCH_SIZE` 到 1024
- 减小 `TRAIN_STEPS` 到 500（快速验证）

### Q3: 信号不准？

- 增加训练步数（`TRAIN_STEPS = 5000`）
- 调整回测参数（更精确的交易成本模型）
- 增加历史数据量（`HISTORY_YEARS = 5`）

### Q4: 如何只关注特定板块？

在 `data_pipeline/data_manager.py` 的过滤逻辑中加入板块筛选：
```python
# 只保留沪深主板 + 创业板
filtered = filtered[filtered['code'].str.match(r'^(00|60|30)')]
```

---

## 十、核心设计保留

以下原版 AlphaGPT 的精髓设计**全部保留并适配到 A 股**：

| 设计亮点 | A 股适配状态 |
|----------|-------------|
| Transformer 公式生成 + StackVM 执行 | ✅ 完全保留 |
| REINFORCE 强化学习训练 | ✅ 完全保留 |
| Looped Transformer（层内循环）| ✅ 完全保留 |
| Newton-Schulz LoRD 正则化 | ✅ 完全保留 |
| QKNorm 稳定注意力 | ✅ 完全保留 |
| Median + MAD 鲁棒归一化 | ✅ 完全保留 |
| 四层解耦架构 | ✅ 完全保留 |

---

**祝投资顺利！**
