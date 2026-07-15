import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'AlphaGPT'))

from model_core.engine import AShareAlphaEngine

if __name__ == '__main__':
    print("=" * 60)
    print("AlphaGPT A-Share Training")
    print("=" * 60)
    
    eng = AShareAlphaEngine(use_lord_regularization=True)
    eng.train()
