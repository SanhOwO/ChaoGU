# AlphaGPT-AShare Mac 交接文档

> **给 Mac 上继续开发的 AI 用**  
> 生成日期：2026-07-13  
> GitHub：https://github.com/SanhOwO/ChaoGU

---

## 一、项目是什么

AlphaGPT 是一个 **AI 自动挖掘量化因子的系统**：

1. Transformer 生成数学公式（token 序列）
2. StackVM 执行公式得到每只股票的分值
3. A 股 T+1 回测引擎评分
4. REINFORCE 强化学习训练生成器
5. 每日收盘后输出 Top 买入/回避信号 → **人工 T+1 日操作**

**不需要任何 API Key**，AKShare 免费数据源。

---

## 二、当前数据状态

```
数据库: AlphaGPT/data_pipeline/ashare_data.db (802MB)
├─ 总记录: 5,174,803 条
├─ 股票数: 3,657 只
├─ 日期范围: 2020-01-02 ~ 2026-07-10
├─ 表名: daily_bars
├─ 列: date, code, open, high, low, close, volume, amount, turnover, market_cap
├─ turnover 覆盖率: 30.8% (1,106 只)
├─ market_cap 覆盖率: 30.8% (1,106 只)
└─ amount 覆盖率: 100% (全部有，由 close × volume 计算)
```

**东方财富 API 被封**：当前环境 IP 已被东方财富拉黑，无法补充 turnover/market_cap。  
**但 factors.py 已做降级兼容**——缺失 turnover 时用 volume 标准化替代，缺失 market_cap 时用 amount 截面排名替代。

---

## 三、已完成的改动（可以直接用）

| 文件 | 说明 |
|------|------|
| `data_pipeline/config.py` | A 股数据配置（SQLite 路径、AKShare 参数） |
| `data_pipeline/db_manager.py` | SQLite 数据库管理 |
| `data_pipeline/data_manager.py` | 数据同步管理器 |
| `model_core/config.py` | 模型配置（A 股交易成本、SQLite 路径） |
| `model_core/vocab.py` | A 股 12 维因子词汇表 |
| `model_core/factors.py` | A 股 12 维因子计算（兼容缺失列） |
| `model_core/backtest.py` | A 股 T+1 回测引擎（涨跌停/印花税） |
| `model_core/data_loader.py` | 从 SQLite 加载数据为 Tensor |
| `model_core/engine.py` | A 股训练引擎（REINFORCE + AShareBacktest） |
| `strategy_manager/config.py` | 策略参数（人工操作阈值） |
| `strategy_manager/signal_generator.py` | 收盘信号生成 + 报告输出 |
| `daily_runner.py` | 每日运行主入口：sync / train / signal / full |
| `csv_to_db.py` | CSV → SQLite 导入（已执行过） |
| `supplement_eastmoney.py` | 东方财富补充 turnover/market_cap（当前被封） |
| `requirements-ashare.txt` | A 股版依赖 |

---

## 四、待办事项（优先级排序）

### 🔴 高优先级 — 必须完成

1. **补充 turnover + market_cap 数据**
   - 当前覆盖率仅 30.8%
   - 在 Mac 上（新 IP）运行 `python supplement_eastmoney.py`
   - 如果东方财富仍然被封，改用 AKShare 的 `stock_zh_a_spot_em` 接口批量获取

2. **运行训练**
   - 命令：`python daily_runner.py --mode train`
   - 产出：`best_meme_strategy.json`（最优因子公式）
   - 注意：训练前确保 turnover/market_cap 尽量补齐

3. **测试信号生成**
   - 命令：`python daily_runner.py --mode signal`
   - 产出：`daily_report_YYYY-MM-DD.txt`
   - 检查报告格式是否正确

### 🟡 中优先级 — 建议完成

4. **选股池优化**
   - 当前是全 A 股 3,657 只
   - 建议改成 **中证1000** 或 **国证2000** 成分股
   - 原因：小盘股波动大、定价效率低，AlphaGPT 更容易挖掘有效因子
   - 修改位置：`data_pipeline/config.py` 或 `data_manager.py` 的过滤逻辑

5. **Dashboard 适配**
   - `dashboard/app.py` 是 Streamlit 看板，但当前是 Meme 币主题
   - 可以改成 A 股版本的可视化看板

6. **回测验证**
   - 运行回测引擎验证因子有效性
   - 检查夏普比率、最大回撤等指标

### 🟢 低优先级 — 可选

7. **Docker 化部署**
8. **定时任务（cron）设置每日自动运行**
9. **多因子组合策略（不只用一个最优公式）**

---

## 五、环境依赖

```bash
# Python 3.10+ (推荐 3.11)
pip install -r requirements-ashare.txt
```

核心依赖：
- `torch>=2.0.0` — 模型训练
- `akshare>=1.10.0` — A 股数据（**无需 API Key**）
- `pandas>=2.0.0` — 数据处理
- `loguru>=0.7.0` — 日志
- `tqdm>=4.66.0` — 进度条

---

## 六、关键命令速查

```bash
# 1. 下载/同步历史数据
python daily_runner.py --mode sync

# 2. 补充 turnover + market_cap（新 IP 下尝试）
python supplement_eastmoney.py

# 3. 训练模型
python daily_runner.py --mode train

# 4. 生成每日信号
python daily_runner.py --mode signal

# 5. 一键全量（同步 + 信号）
python daily_runner.py --mode full
```

---

## 七、已知问题

| 问题 | 状态 | 说明 |
|------|------|------|
| 东方财富 API 被封 | 当前环境 | IP 被拉黑，换 Mac/新 IP 后可试 |
| turnover/market_cap 30.8% | 已兼容 | factors.py 有降级方案，不影响训练 |
| 模型未训练 | 待执行 | 需要运行 `--mode train` |
| 选股池太大 | 建议优化 | 全 A 股 3,657 只，建议缩小到中证1000 |

---

## 八、核心设计（全部保留）

| 设计 | 状态 |
|------|------|
| Transformer 公式生成 + StackVM 执行 | ✅ |
| REINFORCE 强化学习 | ✅ |
| Looped Transformer | ✅ |
| Newton-Schulz LoRD 正则化 | ✅ |
| QKNorm | ✅ |
| Median + MAD 归一化 | ✅ |
| 四层解耦架构 | ✅ |

---

## 九、交接重点提醒

1. ** turnover + market_cap 是第一要务** — 当前 30.8% 覆盖率虽然能跑，但因子质量会受影响
2. **选股池建议改成中证1000** — AlphaGPT 更适合高波动小盘股
3. **训练需要 GPU** — Mac M1/M2/M3 可以用 MPS，否则 CPU 很慢
4. **不需要任何 API Key** — 全部免费数据源
5. **交易是人工操作** — AI 只出信号报告，不自动下单

---

*文档由 Kimi 生成，用于 Mac 环境继续开发。*
