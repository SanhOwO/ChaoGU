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


@dataclass(frozen=True)
class FormulaVocab:
    feature_names: tuple[str, ...]
    operator_names: tuple[str, ...]

    @property
    def feature_count(self) -> int:
        return len(self.feature_names)

    @property
    def operator_offset(self) -> int:
        return self.feature_count

    @property
    def token_names(self) -> tuple[str, ...]:
        return self.feature_names + self.operator_names

    @property
    def size(self) -> int:
        return len(self.token_names)


FORMULA_VOCAB = FormulaVocab(
    feature_names=FEATURE_NAMES,
    operator_names=tuple(cfg[0] for cfg in OPS_CONFIG),
)
