#!/usr/bin/env python3
"""
Quick test of geometry learner on interval problem
"""

import sys
import os
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score

# Add paths
_current_file = os.path.abspath(__file__)
_geometry0_dir = os.path.dirname(_current_file)
_smt_ilp_dir = os.path.dirname(_geometry0_dir)  # SMT-ILP root
sys.path.insert(0, _geometry0_dir)  # Current directory (geometry0)
sys.path.insert(0, _smt_ilp_dir)  # SMT-ILP root

# Find PyGol root
def _find_pygol_root():
    """Find PyGol root directory by checking common locations."""
    if 'PYGOL_ROOT' in os.environ:
        pygol_root = os.environ['PYGOL_ROOT']
        if os.path.exists(pygol_root) and os.path.exists(os.path.join(pygol_root, 'pygol.so')):
            return pygol_root
    # Try as sibling of SMT-ILP
    pygol_root = os.path.join(os.path.dirname(_smt_ilp_dir), 'PyGol')
    if os.path.exists(pygol_root) and os.path.exists(os.path.join(pygol_root, 'pygol.so')):
        return pygol_root
    # Try going up more levels
    for _ in range(3):
        _smt_ilp_dir = os.path.dirname(_smt_ilp_dir)
        pygol_root = os.path.join(_smt_ilp_dir, 'PyGol')
        if os.path.exists(pygol_root) and os.path.exists(os.path.join(pygol_root, 'pygol.so')):
            return pygol_root
    # Check if already in sys.path
    for path in sys.path:
        if os.path.exists(path) and os.path.exists(os.path.join(path, 'pygol.so')):
            return path
    return None

_pygol_root = _find_pygol_root()
if _pygol_root:
    sys.path.insert(0, _pygol_root)

from iterative_pygol_z3_learner_geometry0 import IterativePyGolZ3Learner
from load_geometry0_data import load_interval_data

def test_interval_quick():
    """Quick test on interval problem with small dataset"""
    print("Quick test: Geometry interval problem...")
    
    # Load data from data folder
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    examples_file = os.path.join(data_dir, 'interval_examples.pl')
    X, y, _ = load_interval_data(examples_file)
    
    print(f"\nLoaded {len(X)} examples from data folder")
    
    print(f"\nDataset: {len(X)} examples ({y.sum()} positive, {len(y) - y.sum()} negative)")
    print(f"Features: {list(X.columns)}")
    print(f"X range: [{X['x'].min():.2f}, {X['x'].max():.2f}]")
    
    # Create learner with minimal iterations
    learner = IterativePyGolZ3Learner(
        max_iterations=2,  # Just 2 iterations for quick test
        max_literals=3,
        verbose=True,
        pygol_timeout=30,  # 30 second timeout
        convergence_threshold=0.01
    )
    
    # Set dataset type and BK file
    learner.dataset_type = 'geometry'
    learner.target_predicate = 'interval'
    learner.original_bk_file = None  # Learner creates its own BK with arithmetic operations
    learner.dataset_config = {
        'learning_strategy': 'arithmetic',
        'arithmetic_bounds': (-100, 100),
        'use_arithmetic_learning': True,
        'use_distance_learning': False,
    }
    
    print("\nStarting learning...")
    try:
        learner.fit(X, y)
        
        print(f"\nLearned {len(learner.learned_rules)} rules")
        
        if learner.learned_rules:
            print("\nTop 3 rules:")
            for i, rule in enumerate(learner.learned_rules[:3], 1):
                rule_type = rule.get('type', 'unknown')
                print(f"{i}. Type: {rule_type}, Precision: {rule.get('precision', 0):.3f}, "
                      f"Recall: {rule.get('recall', 0):.3f}")
                if rule_type == 'range':
                    print(f"   {rule.get('lower_bound', 0):.4f} < x < {rule.get('upper_bound', 0):.4f}")
                elif rule_type == 'arithmetic_linear':
                    features_list = rule.get('features', [])
                    coeffs = rule.get('coefficients', [])
                    threshold = rule.get('threshold', 0)
                    print(f"   {coeffs[0]}*{features_list[0]} + {coeffs[1]}*{features_list[1]} <= {threshold:.4f}")
                elif rule_type == 'single_feature':
                    print(f"   {rule.get('feature')} {rule.get('operation')} {rule.get('threshold', 0):.4f}")
        
        # Test prediction
        y_pred = learner.predict(X)
        accuracy = accuracy_score(y, y_pred)
        print(f"\nAccuracy: {accuracy:.3f}")
        
        return accuracy > 0.5  # At least better than random
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_interval_quick()
    if success:
        print("\n[PASS] Quick test passed!")
    else:
        print("\n[FAIL] Quick test failed!")

