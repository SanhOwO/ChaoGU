import torch
import sys
sys.path.insert(0, r'G:\学习理财赚钱\量化\AI\AlphaGPT个股策略\AlphaGPT')

from model_core.vm import StackVM
from model_core.vocab import FORMULA_VOCAB
from model_core.data_loader import AShareDataLoader
from model_core.config import ModelConfig

# Load a small slice of data
loader = AShareDataLoader()
loader.load_data(limit_stocks=10)
feat = loader.feat_tensor

vm = StackVM()
print(f"Vocab size: {FORMULA_VOCAB.size}")
print(f"Feature offset: {FORMULA_VOCAB.operator_offset}")
print(f"Features: {FORMULA_VOCAB.feature_names}")
print(f"Operators: {FORMULA_VOCAB.operator_names}")
print(f"Feat tensor shape: {feat.shape}")

# Test some simple formulas
test_formulas = [
    [0],                          # RET
    [0, 1],                       # RET, TURN (2 items on stack -> invalid)
    [0, 1, 12],                   # RET, TURN, ADD
    [0, 1, 13],                   # RET, TURN, SUB
    [0, 0, 12],                   # RET, RET, ADD
    [1, 2, 3, 12, 12],            # TURN, VOLAT, MOM, ADD, ADD
    [0, 1, 2, 3, 12, 13, 14],     # complex
    [5, 6, 12],                   # RSI, DEV, ADD
]

for tokens in test_formulas:
    res = vm.execute(tokens, feat)
    status = "OK" if res is not None else "FAIL"
    print(f"Tokens {tokens:>30} -> {status}")

# Random test: sample 100 random formulas of length 3-5
import random
random.seed(42)
ok_count = 0
for _ in range(100):
    length = random.randint(3, 5)
    tokens = [random.randint(0, FORMULA_VOCAB.size - 1) for _ in range(length)]
    res = vm.execute(tokens, feat)
    if res is not None:
        ok_count += 1
print(f"\nRandom formulas (100 samples, length 3-5): {ok_count} valid ({ok_count}%)")

# Test with proper polish notation: always push features, then operators
# A valid formula has exactly (operators + 1) features
def generate_valid_formula(max_len=14):
    tokens = []
    stack_needed = 0
    for _ in range(max_len):
        if stack_needed < 1:
            # Must push a feature
            token = random.randint(0, FORMULA_VOCAB.operator_offset - 1)
            tokens.append(token)
            stack_needed += 1
        else:
            # 50% push feature, 50% apply operator
            if random.random() < 0.5 and stack_needed >= 2:
                # Apply operator (needs 2 args)
                op_tokens = [i for i in range(FORMULA_VOCAB.operator_offset, FORMULA_VOCAB.size)
                             if FORMULA_VOCAB.operator_names[i - FORMULA_VOCAB.operator_offset] in ['ADD', 'SUB', 'MUL', 'DIV']]
                token = random.choice(op_tokens)
                tokens.append(token)
                stack_needed -= 1  # 2 items -> 1 result
            else:
                token = random.randint(0, FORMULA_VOCAB.operator_offset - 1)
                tokens.append(token)
                stack_needed += 1
    # End with operators to reduce stack to 1
    while stack_needed > 1:
        op_tokens = [i for i in range(FORMULA_VOCAB.operator_offset, FORMULA_VOCAB.size)
                     if FORMULA_VOCAB.operator_names[i - FORMULA_VOCAB.operator_offset] in ['ADD', 'SUB', 'MUL', 'DIV']]
        if len(tokens) >= max_len:
            break
        token = random.choice(op_tokens)
        tokens.append(token)
        stack_needed -= 1
    return tokens

ok_count = 0
for _ in range(100):
    tokens = generate_valid_formula(14)
    res = vm.execute(tokens, feat)
    if res is not None:
        ok_count += 1
print(f"Constrained valid formulas (100 samples): {ok_count} valid ({ok_count}%)")
