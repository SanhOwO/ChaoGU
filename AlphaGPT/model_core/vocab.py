from dataclasses import dataclass
from .ops import OPS_CONFIG


FEATURE_NAMES = (
    "RET",          # 对数收益率
    "TURN",         # 换手率（标准化）
    "VOLAT",        # 20日波动率
    "MOM",          # 20日动量
    "LOG_AMT",      # 对数成交额
    "RSI",          # 14日 RSI
    "DEV",          # 偏离 20 日均线
    "VR",           # 量比（成交量/MA5成交量）
    "LOG_MC",       # 对数流通市值
    "HL_RANGE",     # 振幅
    "CLOSE_POS",    # 收盘价在高低区间位置
    "MACD_HIST",    # MACD 柱状图
)

# 常数 token：允许模型生成极端值，从而表达“不交易”或“全交易”
CONSTANT_NAMES = (
    "CONST_NEG10",  # -10  常数（sigmoid后≈0，几乎不交易）
    "CONST_NEG5",   # -5   常数（sigmoid后≈0.007，极少交易）
    "CONST_0",      # 0    常数（sigmoid后=0.5，中等交易）
    "CONST_5",      # 5    常数（sigmoid后≈0.993，几乎全交易）
    "CONST_10",     # 10   常数（sigmoid后≈1，全交易）
)

CONSTANT_VALUES = {
    "CONST_NEG10": -10.0,
    "CONST_NEG5": -5.0,
    "CONST_0": 0.0,
    "CONST_5": 5.0,
    "CONST_10": 10.0,
}


@dataclass(frozen=True)
class FormulaVocab:
    feature_names: tuple[str, ...]
    operator_names: tuple[str, ...]
    constant_names: tuple[str, ...]

    @property
    def feature_count(self) -> int:
        return len(self.feature_names)

    @property
    def operator_offset(self) -> int:
        return self.feature_count

    @property
    def constant_offset(self) -> int:
        return self.feature_count + len(self.operator_names)

    @property
    def token_names(self) -> tuple[str, ...]:
        return self.feature_names + self.operator_names + self.constant_names

    @property
    def size(self) -> int:
        return len(self.token_names)

    def get_constant_value(self, token_id: int) -> float | None:
        """根据 token_id 返回常数值，若不是常数 token 则返回 None"""
        offset = self.constant_offset
        idx = token_id - offset
        if 0 <= idx < len(self.constant_names):
            return CONSTANT_VALUES[self.constant_names[idx]]
        return None


FORMULA_VOCAB = FormulaVocab(
    feature_names=FEATURE_NAMES,
    operator_names=tuple(cfg[0] for cfg in OPS_CONFIG),
    constant_names=CONSTANT_NAMES,
)
