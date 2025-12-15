#!/usr/bin/env python3
"""
Test geometry learner on all geometry problems separately:
1. Interval problem
2. Halfplane problem
"""

import sys
import os
import time
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

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
    parent = _smt_ilp_dir
    for _ in range(3):
        parent = os.path.dirname(parent)
        pygol_root = os.path.join(parent, 'PyGol')
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
from load_geometry0_data import load_interval_data, load_halfplane_data

def test_interval_problem():
    """Test interval problem"""
    problem_start_time = time.time()
    print("\nTesting interval problem...")
    
    # Load data
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    examples_file = os.path.join(data_dir, 'interval_examples.pl')
    # Note: BK file is created by the learner, not needed here
    X, y, _ = load_interval_data(examples_file)
    
    # Get random seed from environment variable (set by run_all_pygol_z3.py for multiple trials)
    # Default to 96 for backward compatibility
    random_seed = int(os.environ.get('PYGOL_RANDOM_SEED', '96'))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=random_seed)
    
    # Train on training set
    # Increased iterations and timeout for interval to learn more rules
    learner_train = IterativePyGolZ3Learner(
        max_iterations=5,  # Increased from 3 to 5 for more rules
        max_literals=3,
        verbose=False,
        pygol_timeout=45,  # Increased from 30 to 45 for more time to learn
        convergence_threshold=0.005  # Lower threshold to continue learning
    )
    learner_train.dataset_type = 'geometry'
    learner_train.target_predicate = 'interval'
    # Note: original_bk_file not needed - learner creates its own BK with arithmetic operations
    learner_train.original_bk_file = None
    learner_train.dataset_config = {
        'learning_strategy': 'arithmetic',
        'arithmetic_bounds': (-100, 100),
        'use_arithmetic_learning': True,
        'use_distance_learning': False,
    }
    learner_train.fit(X_train, y_train)
    
    # Evaluate on both train and test
    y_pred_train = learner_train.predict(X_train)
    y_pred_test = learner_train.predict(X_test)
    
    train_accuracy = accuracy_score(y_train, y_pred_train)
    test_accuracy = accuracy_score(y_test, y_pred_test)
    train_precision = precision_score(y_train, y_pred_train, zero_division=0)
    test_precision = precision_score(y_test, y_pred_test, zero_division=0)
    train_recall = recall_score(y_train, y_pred_train, zero_division=0)
    test_recall = recall_score(y_test, y_pred_test, zero_division=0)
    train_f1 = f1_score(y_train, y_pred_train, zero_division=0)
    test_f1 = f1_score(y_test, y_pred_test, zero_division=0)
    
    
    return {
        'problem': 'Interval',
        'num_rules': len(learner_train.learned_rules),
        'train_accuracy': train_accuracy,
        'test_accuracy': test_accuracy,
        'train_precision': train_precision,
        'test_precision': test_precision,
        'train_recall': train_recall,
        'test_recall': test_recall,
        'train_f1': train_f1,
        'test_f1': test_f1,
        'pygol_learned': len(learner_train.learned_rules) > 0,
        'accuracy_pass': test_accuracy > 0.7,
        'time': time.time() - problem_start_time
    }


def test_halfplane_problem():
    """Test halfplane problem"""
    problem_start_time = time.time()
    print("\nTesting halfplane problem...")
    
    # Load data
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    examples_file = os.path.join(data_dir, 'halfplane_examples.pl')
    # Note: BK file is created by the learner, not needed here
    X, y, _ = load_halfplane_data(examples_file)
    
    # Get random seed from environment variable (set by run_all_pygol_z3.py for multiple trials)
    # Default to 96 for backward compatibility
    random_seed = int(os.environ.get('PYGOL_RANDOM_SEED', '96'))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=random_seed)
    
    # Train on training set
    learner_train = IterativePyGolZ3Learner(
        max_iterations=5,  # Increased from 3 to 5 for consistency
        max_literals=3,
        verbose=False,
        pygol_timeout=45,  # Increased from 30 to 45 for consistency
        convergence_threshold=0.005  # Lower threshold to continue learning
    )
    learner_train.dataset_type = 'geometry'
    learner_train.target_predicate = 'halfplane'
    # Note: original_bk_file not needed - learner creates its own BK with arithmetic operations
    learner_train.original_bk_file = None
    learner_train.dataset_config = {
        'learning_strategy': 'arithmetic',
        'arithmetic_bounds': (-100, 100),
        'use_arithmetic_learning': True,
        'use_distance_learning': False,
    }
    learner_train.fit(X_train, y_train)
    
    # Evaluate on both train and test
    y_pred_train = learner_train.predict(X_train)
    y_pred_test = learner_train.predict(X_test)
    
    train_accuracy = accuracy_score(y_train, y_pred_train)
    test_accuracy = accuracy_score(y_test, y_pred_test)
    train_precision = precision_score(y_train, y_pred_train, zero_division=0)
    test_precision = precision_score(y_test, y_pred_test, zero_division=0)
    train_recall = recall_score(y_train, y_pred_train, zero_division=0)
    test_recall = recall_score(y_test, y_pred_test, zero_division=0)
    train_f1 = f1_score(y_train, y_pred_train, zero_division=0)
    test_f1 = f1_score(y_test, y_pred_test, zero_division=0)
    
    
    return {
        'problem': 'Halfplane',
        'num_rules': len(learner_train.learned_rules),
        'train_accuracy': train_accuracy,
        'test_accuracy': test_accuracy,
        'train_precision': train_precision,
        'test_precision': test_precision,
        'train_recall': train_recall,
        'test_recall': test_recall,
        'train_f1': train_f1,
        'test_f1': test_f1,
        'pygol_learned': len(learner_train.learned_rules) > 0,
        'accuracy_pass': test_accuracy > 0.7,
        'time': time.time() - problem_start_time
    }


if __name__ == "__main__":
    # Data directory
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    
    # Check if data exists, if not generate it
    if not os.path.exists(os.path.join(data_dir, 'interval_examples.pl')):
        print("Data files not found. Generating...")
        import subprocess
        subprocess.run([sys.executable, os.path.join(os.path.dirname(__file__), 'generate_geometry0.py')])
    
    print("Testing all geometry0 problems...")
    
    results = []
    
    # Test Interval
    try:
        result = test_interval_problem()
        results.append(result)
    except Exception as e:
        print(f"\n[FAIL] Interval test failed: {e}")
        import traceback
        traceback.print_exc()
        results.append({
            'problem': 'Interval',
            'num_rules': 0,
            'train_accuracy': 0.0,
            'test_accuracy': 0.0,
            'train_f1': 0.0,
            'test_f1': 0.0,
            'pygol_learned': False,
            'accuracy_pass': False
        })
    
    # Test Halfplane
    try:
        result = test_halfplane_problem()
        results.append(result)
    except Exception as e:
        print(f"\n[FAIL] Halfplane test failed: {e}")
        import traceback
        traceback.print_exc()
        results.append({
            'problem': 'Halfplane',
            'num_rules': 0,
            'train_accuracy': 0.0,
            'test_accuracy': 0.0,
            'train_f1': 0.0,
            'test_f1': 0.0,
            'pygol_learned': False,
            'accuracy_pass': False
        })
    
    # Summary
    print("\nSummary:")
    
    if results:
        print(f"\n{'Problem':<25} {'Rules':<8} {'Train Acc':<12} {'Test Acc':<12} {'Test F1':<12} {'Time (s)':<10} {'Status':<10}")
        print("-" * 100)
        for r in results:
            status = "[PASS]" if (r['accuracy_pass'] and r['pygol_learned']) else "[FAIL]"
            time_str = f"{r.get('time', 0.0):.2f}" if r.get('time') is not None else "N/A"
            print(f"{r['problem']:<25} {r['num_rules']:<8} {r['train_accuracy']:<12.4f} {r['test_accuracy']:<12.4f} {r['test_f1']:<12.4f} {time_str:<10} {status:<10}")
        
        avg_test_acc = np.mean([r['test_accuracy'] for r in results])
        avg_test_f1 = np.mean([r['test_f1'] for r in results])
        print(f"\nAverage Test Accuracy: {avg_test_acc:.4f}")
        print(f"Average Test F1: {avg_test_f1:.4f}")
    
        all_passed = all(r['accuracy_pass'] and r['pygol_learned'] for r in results)
        if all_passed:
            print("\nAll tests passed")
        else:
            print("\nSome tests failed")
        
        # When run as part of batch experiments (PYGOL_RANDOM_SEED set), always exit 0
        # to allow result collection. Exit code is only meaningful when running standalone.
        if os.environ.get('PYGOL_RANDOM_SEED'):
            sys.exit(0)
        else:
            sys.exit(0 if all_passed else 1)
    else:
        print("\nNo results to summarize.")
        # Allow batch runs to continue even with no results
        if os.environ.get('PYGOL_RANDOM_SEED'):
            sys.exit(0)
        else:
            sys.exit(1)

