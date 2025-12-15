#!/usr/bin/env python3
"""
Test geometry learner on all geometry1 problems:
1. 3D Halfplane problem
2. Conjunction problem
3. Multiple Halfplanes problem
4. 3D Interval problem
"""

import sys
import os
import time
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

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
from load_geometry1_data import (
    load_halfplane3d_data,
    load_conjunction_data,
    load_multihalfplane_data,
    load_interval3d_data
)


def test_problem(problem_name, load_func, examples_file, bk_file, target_predicate):
    """Test a single geometry1 problem"""
    problem_start_time = time.time()
    print(f"\nTesting {problem_name}...")
    
    # Load data
    X, y, _ = load_func(examples_file, bk_file)
    
    # Get random seed from environment variable (set by run_all_pygol_z3.py for multiple trials)
    # Default to 42 for backward compatibility
    random_seed = int(os.environ.get('PYGOL_RANDOM_SEED', '42'))
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=random_seed, stratify=y
    )
    
    # Create learner
    # For geometry problems, use longer timeout and more literals to help PyGol learn structure
    learner = IterativePyGolZ3Learner(
        max_iterations=10,
        max_literals=6,  # Increased from 5 to allow arithmetic rules (need 5-6 literals)
        verbose=True,
        pygol_timeout=120,  # Increased from 60 to allow PyGol to explore arithmetic predicates
        convergence_threshold=0.001
    )
    
    learner.dataset_type = 'geometry'
    learner.target_predicate = target_predicate
    learner.original_bk_file = bk_file
    learner.dataset_config = {
        'learning_strategy': 'arithmetic',
        'arithmetic_bounds': (-100, 100),
        'use_arithmetic_learning': True,
        'use_distance_learning': False,
    }
    
    # Train
    try:
        learner.fit(X_train, y_train)
    except Exception as e:
        print(f"Error: Learning failed: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # Evaluate on training set
    y_train_pred = learner.predict(X_train)
    train_accuracy = accuracy_score(y_train, y_train_pred)
    train_precision = precision_score(y_train, y_train_pred, zero_division=0)
    train_recall = recall_score(y_train, y_train_pred, zero_division=0)
    train_f1 = f1_score(y_train, y_train_pred, zero_division=0)
    
    # Evaluate on test set
    y_test_pred = learner.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_precision = precision_score(y_test, y_test_pred, zero_division=0)
    test_recall = recall_score(y_test, y_test_pred, zero_division=0)
    test_f1 = f1_score(y_test, y_test_pred, zero_division=0)
    
    if learner.learned_rules:
        print(f"Rules learned: {len(learner.learned_rules)}")
        print(f"Train accuracy: {train_accuracy:.4f}, Test accuracy: {test_accuracy:.4f}")
    
    return {
        'problem': problem_name,
        'num_rules': len(learner.learned_rules),
        'train_accuracy': train_accuracy,
        'test_accuracy': test_accuracy,
        'train_precision': train_precision,
        'test_precision': test_precision,
        'train_recall': train_recall,
        'test_recall': test_recall,
        'train_f1': train_f1,
        'test_f1': test_f1,
        'pygol_learned': len(learner.learned_rules) > 0,
        'accuracy_pass': test_accuracy > 0.65,  # Lower threshold for geometry1 problems (they're more difficult)
        'time': time.time() - problem_start_time
    }


if __name__ == "__main__":
    # Data directory
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    
    # Check if data exists, if not generate it
    if not os.path.exists(os.path.join(data_dir, 'halfplane3d_examples.pl')):
        print("Data files not found. Generating...")
        import subprocess
        subprocess.run([sys.executable, os.path.join(os.path.dirname(__file__), 'generate_geometry1.py')])
    
    print("Testing all geometry1 problems...")
    
    results = []
    
    # Test 1: 3D Halfplane
    try:
        result = test_problem(
            '3D Halfplane',
            load_halfplane3d_data,
            os.path.join(data_dir, 'halfplane3d_examples.pl'),
            os.path.join(data_dir, 'halfplane3d_BK.pl'),
            'halfplane3d'
        )
        if result:
            results.append(result)
    except Exception as e:
        print(f"Error: 3D Halfplane test failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 2: Conjunction
    try:
        result = test_problem(
            'Conjunction',
            load_conjunction_data,
            os.path.join(data_dir, 'conjunction_examples.pl'),
            os.path.join(data_dir, 'conjunction_BK.pl'),
            'conjunction'
        )
        if result:
            results.append(result)
    except Exception as e:
        print(f"Error: Conjunction test failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 3: Multiple Halfplanes
    try:
        result = test_problem(
            'Multiple Halfplanes',
            load_multihalfplane_data,
            os.path.join(data_dir, 'multihalfplane_examples.pl'),
            os.path.join(data_dir, 'multihalfplane_BK.pl'),
            'multihalfplane'
        )
        if result:
            results.append(result)
    except Exception as e:
        print(f"Error: Multiple Halfplanes test failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 4: 3D Interval
    try:
        result = test_problem(
            '3D Interval',
            load_interval3d_data,
            os.path.join(data_dir, 'interval3d_examples.pl'),
            os.path.join(data_dir, 'interval3d_BK.pl'),
            'interval3d'
        )
        if result:
            results.append(result)
    except Exception as e:
        print(f"Error: 3D Interval test failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Summary
    print("\nSummary:")
    
    if results:
        print(f"\n{'Problem':<25} {'Rules':<8} {'Train Acc':<12} {'Test Acc':<12} {'Test F1':<12} {'Time (s)':<10} {'Status':<10}")
        print("-" * 100)
        for r in results:
            status = "PASS" if (r['accuracy_pass'] and r['pygol_learned']) else "FAIL"
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
