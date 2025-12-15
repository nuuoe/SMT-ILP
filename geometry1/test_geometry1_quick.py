#!/usr/bin/env python3
"""
Quick test of geometry1 learner on one problem (3D Halfplane)
"""

import sys
import os
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Add paths
_current_file = os.path.abspath(__file__)
_geometry1_dir = os.path.dirname(_current_file)
_smt_ilp_dir = os.path.dirname(_geometry1_dir)  # SMT-ILP root
sys.path.insert(0, _geometry1_dir)  # Current directory (geometry1)
sys.path.insert(0, _smt_ilp_dir)  # SMT-ILP root

# Find PyGol root
def _find_pygol_root():
    """Find PyGol root directory by checking common locations."""
    if 'PYGOL_ROOT' in os.environ:
        pygol_root = os.environ['PYGOL_ROOT']
        if os.path.exists(pygol_root) and os.path.exists(os.path.join(pygol_root, 'pygol.so')):
            return pygol_root
    pygol_root = os.path.join(os.path.dirname(_smt_ilp_dir), 'PyGol')
    if os.path.exists(pygol_root) and os.path.exists(os.path.join(pygol_root, 'pygol.so')):
        return pygol_root
    parent = _smt_ilp_dir
    for _ in range(3):
        parent = os.path.dirname(parent)
        pygol_root = os.path.join(parent, 'PyGol')
        if os.path.exists(pygol_root) and os.path.exists(os.path.join(pygol_root, 'pygol.so')):
            return pygol_root
    for path in sys.path:
        if os.path.exists(path) and os.path.exists(os.path.join(path, 'pygol.so')):
            return path
    return None

_pygol_root = _find_pygol_root()
if _pygol_root:
    sys.path.insert(0, _pygol_root)

from iterative_pygol_z3_learner_geometry1 import IterativePyGolZ3Learner
from load_geometry1_data import load_halfplane3d_data

def test_halfplane3d_quick():
    """Quick test on 3D halfplane problem with small dataset"""
    print("Quick test: Geometry1 3D Halfplane problem")
    
    # Load data from data folder
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    examples_file = os.path.join(data_dir, 'halfplane3d_examples.pl')
    bk_file = os.path.join(data_dir, 'halfplane3d_BK.pl')
    
    # Check if data exists, if not generate it
    if not os.path.exists(examples_file):
        print("Data files not found. Generating...")
        import subprocess
        subprocess.run([sys.executable, os.path.join(os.path.dirname(__file__), 'generate_geometry1.py')])
    
    X, y, _ = load_halfplane3d_data(examples_file, bk_file)
    
    # Use smaller subset for quick test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Use only first 20 examples for quick test
    X_train = X_train.head(20)
    y_train = y_train[:20]
    
    print(f"\nDataset: {len(X_train)} training examples ({y_train.sum()} positive, {len(y_train) - y_train.sum()} negative)")
    print(f"Features: {list(X_train.columns)}")
    for col in X_train.columns:
        print(f"  {col}: [{X_train[col].min():.2f}, {X_train[col].max():.2f}]")
    
    # Create learner with minimal iterations
    learner = IterativePyGolZ3Learner(
        max_iterations=2,  # Just 2 iterations for quick test
        max_literals=5,
        verbose=True,
        pygol_timeout=30,  # 30 second timeout
        convergence_threshold=0.01
    )
    
    # Set dataset type and BK file
    learner.dataset_type = 'geometry'
    learner.target_predicate = 'halfplane3d'
    learner.original_bk_file = bk_file
    learner.dataset_config = {
        'learning_strategy': 'arithmetic',
        'arithmetic_bounds': (-100, 100),
        'use_arithmetic_learning': True,
        'use_distance_learning': False,
    }
    
    print("\nStarting learning...")
    try:
        learner.fit(X_train, y_train)
        
        print(f"\nLearned {len(learner.learned_rules)} rules")
        
        if learner.learned_rules:
            print("\nTop 3 rules:")
            for i, rule in enumerate(learner.learned_rules[:3], 1):
                rule_type = rule.get('type', 'unknown')
                print(f"{i}. Type: {rule_type}, Precision: {rule.get('precision', 0):.3f}, "
                      f"Recall: {rule.get('recall', 0):.3f}")
                if rule_type == 'arithmetic_linear':
                    features_list = rule.get('features', [])
                    coeffs = rule.get('coefficients', [])
                    threshold = rule.get('threshold', 0)
                    if len(coeffs) == 3:
                        print(f"   {coeffs[0]:.2f}*{features_list[0]} + {coeffs[1]:.2f}*{features_list[1]} + {coeffs[2]:.2f}*{features_list[2]} <= {threshold:.4f}")
        
        # Test prediction on training set
        y_pred = learner.predict(X_train)
        accuracy = accuracy_score(y_train, y_pred)
        print(f"\nTraining Accuracy: {accuracy:.3f}")
        
        return accuracy > 0.5  # At least better than random
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_halfplane3d_quick()
    if success:
        print("\n[PASS] Quick test passed!")
    else:
        print("\n[FAIL] Quick test failed!")
