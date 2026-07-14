import torch
from .ops import OPS_CONFIG
from .vocab import FORMULA_VOCAB

class StackVM:
    def __init__(self):
        self.feat_offset = FORMULA_VOCAB.operator_offset
        self.op_map = {i + self.feat_offset: cfg[1] for i, cfg in enumerate(OPS_CONFIG)}
        self.arity_map = {i + self.feat_offset: cfg[2] for i, cfg in enumerate(OPS_CONFIG)}
        self.const_offset = FORMULA_VOCAB.constant_offset

    def execute(self, formula_tokens, feat_tensor):
        stack = []
        try:
            for token in formula_tokens:
                token = int(token)
                if token < self.feat_offset:
                    if token >= feat_tensor.shape[1]:
                        return None
                    stack.append(feat_tensor[:, token, :])
                elif token in self.op_map:
                    arity = self.arity_map[token]
                    if len(stack) < arity: return None
                    args = []
                    for _ in range(arity):
                        args.append(stack.pop())
                    args.reverse()
                    func = self.op_map[token]
                    res = func(*args)
                    if torch.isnan(res).any() or torch.isinf(res).any():
                        res = torch.nan_to_num(res, nan=0.0, posinf=1.0, neginf=-1.0)
                    stack.append(res)
                elif token >= self.const_offset:
                    # 常数 token：推入一个与 feat_tensor 同形状的常数张量
                    const_val = FORMULA_VOCAB.get_constant_value(token)
                    if const_val is None:
                        return None
                    const_tensor = torch.full(
                        (feat_tensor.shape[0], feat_tensor.shape[2]),
                        const_val,
                        dtype=feat_tensor.dtype,
                        device=feat_tensor.device
                    )
                    stack.append(const_tensor)
                else:
                    return None
            if len(stack) == 1:
                return stack[0]
            else:
                return None
        except Exception:
            return None
