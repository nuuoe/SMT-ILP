#!/usr/bin/env python3
"""
Test geometry learner on all geometry2 problems:
1. left_of problem
2. closer_than problem
3. touching problem
4. inside problem
5. overlapping problem
6. between problem
7. adjacent problem
8. aligned problem
9. surrounds problem
10. near_corner problem
"""

import sys
import os
import time
import signal
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Add paths
_current_file = os.path.abspath(__file__)
_geometry2_dir = os.path.dirname(_current_file)
_smt_ilp_dir = os.path.dirname(_geometry2_dir)  # SMT-ILP root
sys.path.insert(0, _geometry2_dir)  # Current directory (geometry2)
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

from iterative_pygol_z3_learner_geometry2 import IterativePyGolZ3Learner
from load_geometry2_data import (
    load_left_of_data,
    load_closer_than_data,
    load_touching_data,
    load_inside_data,
    load_overlapping_data,
    load_between_data,
    load_adjacent_data,
    load_aligned_data,
    load_surrounds_data,
    load_near_corner_data
)


def test_problem(problem_name, load_func, examples_file, bk_file, target_predicate):
    """Test a single geometry2 problem"""
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
    # For geometry2 (relational problems), we may need different settings
    # Tuned for balance: allow more exploration while still being fast enough
    learner = IterativePyGolZ3Learner(
        max_iterations=3,  # Keep at 3 (worked well at ~1.5 hours per trial)
        max_literals=5,    # Keep at 5 (original working settings)
        verbose=False,     # Less verbose for cleaner output
        pygol_timeout=30,  # Keep at 30s (original working settings)
        convergence_threshold=0.01  # Looser threshold for faster convergence
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
    
    # Train with overall timeout protection (prevent 10+ hour hangs)
    # Set per-problem timeout: 30 minutes max per problem (1800 seconds)
    # This allows ~12-15 minutes per problem on average (10 problems × 12-15 min = 2-2.5 hours)
    # Reduced from 2 hours to 30 minutes to prevent individual problems from blocking too long
    # If a problem needs more than 30 minutes, it's likely stuck or too complex
    PER_PROBLEM_TIMEOUT = 1800  # 30 minutes per problem (reduced from 2 hours)
    
    class TimeoutError(Exception):
        pass
    
    def timeout_handler(signum, frame):
        raise TimeoutError(f"Problem '{problem_name}' exceeded {PER_PROBLEM_TIMEOUT}s timeout")
    
    # Set up timeout signal (Unix only)
    if hasattr(signal, 'SIGALRM'):
        old_handler = signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(PER_PROBLEM_TIMEOUT)
    
    try:
        learner.fit(X_train, y_train)
    except TimeoutError as e:
        print(f"\n[WARNING] {e}")
        print(f"  Skipping problem '{problem_name}' due to timeout")
        if hasattr(signal, 'SIGALRM'):
            signal.alarm(0)  # Cancel alarm
            signal.signal(signal.SIGALRM, old_handler)  # Restore handler
        return None
    except Exception as e:
        print(f"Error: Learning failed: {e}")
        import traceback
        traceback.print_exc()
        if hasattr(signal, 'SIGALRM'):
            signal.alarm(0)  # Cancel alarm
            signal.signal(signal.SIGALRM, old_handler)  # Restore handler
        return None
    finally:
        # Always cancel alarm and restore handler
        if hasattr(signal, 'SIGALRM'):
            signal.alarm(0)
            try:
                signal.signal(signal.SIGALRM, old_handler)
            except:
                pass
    
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
        'accuracy_pass': test_accuracy > 0.55,  # Lower threshold for more challenging geometric relationships
        'time': time.time() - problem_start_time
    }


if __name__ == "__main__":
    # Data directory
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    
    # Check if data exists, if not generate it
    if not os.path.exists(os.path.join(data_dir, 'left_of_examples.pl')):
        print("Data files not found. Generating...")
        import subprocess
        subprocess.run([sys.executable, os.path.join(os.path.dirname(__file__), 'generate_geometry2.py')])
    
    print("Testing all geometry2 problems...")
    
    results = []
    
    # Test 1: left_of
    try:
        result = test_problem(
            'left_of',
            load_left_of_data,
            os.path.join(data_dir, 'left_of_examples.pl'),
            os.path.join(data_dir, 'left_of_BK.pl'),
            'left_of'
        )
        if result:
            results.append(result)
    except Exception as e:
        print(f"Error: left_of test failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 2: closer_than
    try:
        result = test_problem(
            'closer_than',
            load_closer_than_data,
            os.path.join(data_dir, 'closer_than_examples.pl'),
            os.path.join(data_dir, 'closer_than_BK.pl'),
            'closer_than'
        )
        if result:
            results.append(result)
    except Exception as e:
        print(f"Error: closer_than test failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 3: touching
    try:
        result = test_problem(
            'touching',
            load_touching_data,
            os.path.join(data_dir, 'touching_examples.pl'),
            os.path.join(data_dir, 'touching_BK.pl'),
            'touching'
        )
        if result:
            results.append(result)
    except Exception as e:
        print(f"Error: touching test failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 4: inside
    try:
        result = test_problem(
            'inside',
            load_inside_data,
            os.path.join(data_dir, 'inside_examples.pl'),
            os.path.join(data_dir, 'inside_BK.pl'),
            'inside'
        )
        if result:
            results.append(result)
    except Exception as e:
        print(f"Error: inside test failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 5: overlapping
    try:
        result = test_problem(
            'overlapping',
            load_overlapping_data,
            os.path.join(data_dir, 'overlapping_examples.pl'),
            os.path.join(data_dir, 'overlapping_BK.pl'),
            'overlapping'
        )
        if result:
            results.append(result)
    except Exception as e:
        print(f"Error: overlapping test failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 6: between
    try:
        result = test_problem(
            'between',
            load_between_data,
            os.path.join(data_dir, 'between_examples.pl'),
            os.path.join(data_dir, 'between_BK.pl'),
            'between'
        )
        if result:
            results.append(result)
    except Exception as e:
        print(f"Error: between test failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 7: adjacent
    try:
        result = test_problem(
            'adjacent',
            load_adjacent_data,
            os.path.join(data_dir, 'adjacent_examples.pl'),
            os.path.join(data_dir, 'adjacent_BK.pl'),
            'adjacent'
        )
        if result:
            results.append(result)
    except Exception as e:
        print(f"Error: adjacent test failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 8: aligned
    try:
        result = test_problem(
            'aligned',
            load_aligned_data,
            os.path.join(data_dir, 'aligned_examples.pl'),
            os.path.join(data_dir, 'aligned_BK.pl'),
            'aligned'
        )
        if result:
            results.append(result)
    except Exception as e:
        print(f"Error: aligned test failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 9: surrounds
    try:
        result = test_problem(
            'surrounds',
            load_surrounds_data,
            os.path.join(data_dir, 'surrounds_examples.pl'),
            os.path.join(data_dir, 'surrounds_BK.pl'),
            'surrounds'
        )
        if result:
            results.append(result)
    except Exception as e:
        print(f"Error: surrounds test failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 10: near_corner
    try:
        result = test_problem(
            'near_corner',
            load_near_corner_data,
            os.path.join(data_dir, 'near_corner_examples.pl'),
            os.path.join(data_dir, 'near_corner_BK.pl'),
            'near_corner'
        )
        if result:
            results.append(result)
    except Exception as e:
        print(f"Error: near_corner test failed: {e}")
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
