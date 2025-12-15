#!/usr/bin/env python3
"""
Test geometry learner on all geometry3 problems:

Category 1: Nonlinear Regions
1. in_circle
2. in_ellipse
3. hyperbola_side
4. xy_less_than
5. quad_strip

Category 2: Disjunctive (OR) Regions
6. union_halfplanes
7. circle_or_box
8. piecewise
9. fallback_region

Category 3: Non-convex / Piecewise / Hybrid Regions
10. donut
11. lshape
12. above_parabola
13. sinusoidal
14. crescent
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
_geometry3_dir = os.path.dirname(_current_file)
_smt_ilp_dir = os.path.dirname(_geometry3_dir)  # SMT-ILP root
sys.path.insert(0, _geometry3_dir)  # Current directory (geometry3)
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

from iterative_pygol_z3_learner_geometry3 import IterativePyGolZ3Learner
from load_geometry3_data import (
    # Category 1: Nonlinear
    load_in_circle_data,
    load_in_ellipse_data,
    load_hyperbola_side_data,
    load_xy_less_than_data,
    load_quad_strip_data,
    # Category 2: Disjunctive
    load_union_halfplanes_data,
    load_circle_or_box_data,
    load_piecewise_data,
    load_fallback_region_data,
    # Category 3: Non-convex
    load_donut_data,
    load_lshape_data,
    load_above_parabola_data,
    load_sinusoidal_data,
    load_crescent_data
)


def test_problem(problem_name, load_func, examples_file, bk_file, target_predicate):
    """Test a single geometry3 problem"""
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
    # For geometry3 (nonlinear/disjunctive), use more iterations and longer timeout
    learner = IterativePyGolZ3Learner(
        max_iterations=5,
        max_literals=6,
        verbose=True,
        pygol_timeout=60,
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
        'accuracy_pass': test_accuracy > 0.55,  # Lower threshold for challenging nonlinear problems
        'time': time.time() - problem_start_time
    }


if __name__ == "__main__":
    # Data directory
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    
    # Check if data exists, if not generate it
    if not os.path.exists(os.path.join(data_dir, 'in_circle_examples.pl')):
        print("Data files not found. Generating...")
        import subprocess
        subprocess.run([sys.executable, os.path.join(os.path.dirname(__file__), 'generate_geometry3.py')])
    
    print("Testing all geometry3 problems...")
    
    results = []
    
    # Category 1: Nonlinear Regions
    print("\nCategory 1: Nonlinear Regions")
    
    try:
        result = test_problem(
            'in_circle',
            load_in_circle_data,
            os.path.join(data_dir, 'in_circle_examples.pl'),
            os.path.join(data_dir, 'in_circle_BK.pl'),
            'in_circle'
        )
        if result:
            results.append(result)
    except Exception as e:
        print(f"Error: in_circle test failed: {e}")
        import traceback
        traceback.print_exc()
    
    try:
        result = test_problem(
            'in_ellipse',
            load_in_ellipse_data,
            os.path.join(data_dir, 'in_ellipse_examples.pl'),
            os.path.join(data_dir, 'in_ellipse_BK.pl'),
            'in_ellipse'
        )
        if result:
            results.append(result)
    except Exception as e:
        print(f"Error: in_ellipse test failed: {e}")
        import traceback
        traceback.print_exc()
    
    try:
        result = test_problem(
            'hyperbola_side',
            load_hyperbola_side_data,
            os.path.join(data_dir, 'hyperbola_side_examples.pl'),
            os.path.join(data_dir, 'hyperbola_side_BK.pl'),
            'hyperbola_side'
        )
        if result:
            results.append(result)
    except Exception as e:
        print(f"Error: hyperbola_side test failed: {e}")
        import traceback
        traceback.print_exc()
    
    try:
        result = test_problem(
            'xy_less_than',
            load_xy_less_than_data,
            os.path.join(data_dir, 'xy_less_than_examples.pl'),
            os.path.join(data_dir, 'xy_less_than_BK.pl'),
            'xy_less_than'
        )
        if result:
            results.append(result)
    except Exception as e:
        print(f"Error: xy_less_than test failed: {e}")
        import traceback
        traceback.print_exc()
    
    try:
        result = test_problem(
            'quad_strip',
            load_quad_strip_data,
            os.path.join(data_dir, 'quad_strip_examples.pl'),
            os.path.join(data_dir, 'quad_strip_BK.pl'),
            'quad_strip'
        )
        if result:
            results.append(result)
    except Exception as e:
        print(f"Error: quad_strip test failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Category 2: Disjunctive (OR) Regions
    print("\nCategory 2: Disjunctive (OR) Regions")
    
    try:
        result = test_problem(
            'union_halfplanes',
            load_union_halfplanes_data,
            os.path.join(data_dir, 'union_halfplanes_examples.pl'),
            os.path.join(data_dir, 'union_halfplanes_BK.pl'),
            'union_halfplanes'
        )
        if result:
            results.append(result)
    except Exception as e:
        print(f"Error: union_halfplanes test failed: {e}")
        import traceback
        traceback.print_exc()
    
    try:
        result = test_problem(
            'circle_or_box',
            load_circle_or_box_data,
            os.path.join(data_dir, 'circle_or_box_examples.pl'),
            os.path.join(data_dir, 'circle_or_box_BK.pl'),
            'circle_or_box'
        )
        if result:
            results.append(result)
    except Exception as e:
        print(f"Error: circle_or_box test failed: {e}")
        import traceback
        traceback.print_exc()
    
    try:
        result = test_problem(
            'piecewise',
            load_piecewise_data,
            os.path.join(data_dir, 'piecewise_examples.pl'),
            os.path.join(data_dir, 'piecewise_BK.pl'),
            'piecewise'
        )
        if result:
            results.append(result)
    except Exception as e:
        print(f"Error: piecewise test failed: {e}")
        import traceback
        traceback.print_exc()
    
    try:
        result = test_problem(
            'fallback_region',
            load_fallback_region_data,
            os.path.join(data_dir, 'fallback_region_examples.pl'),
            os.path.join(data_dir, 'fallback_region_BK.pl'),
            'fallback_region'
        )
        if result:
            results.append(result)
    except Exception as e:
        print(f"Error: fallback_region test failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Category 3: Non-convex / Piecewise / Hybrid Regions
    print("\nCategory 3: Non-convex / Piecewise / Hybrid Regions")
    
    try:
        result = test_problem(
            'donut',
            load_donut_data,
            os.path.join(data_dir, 'donut_examples.pl'),
            os.path.join(data_dir, 'donut_BK.pl'),
            'donut'
        )
        if result:
            results.append(result)
    except Exception as e:
        print(f"Error: donut test failed: {e}")
        import traceback
        traceback.print_exc()
    
    try:
        result = test_problem(
            'lshape',
            load_lshape_data,
            os.path.join(data_dir, 'lshape_examples.pl'),
            os.path.join(data_dir, 'lshape_BK.pl'),
            'lshape'
        )
        if result:
            results.append(result)
    except Exception as e:
        print(f"Error: lshape test failed: {e}")
        import traceback
        traceback.print_exc()
    
    try:
        result = test_problem(
            'above_parabola',
            load_above_parabola_data,
            os.path.join(data_dir, 'above_parabola_examples.pl'),
            os.path.join(data_dir, 'above_parabola_BK.pl'),
            'above_parabola'
        )
        if result:
            results.append(result)
    except Exception as e:
        print(f"Error: above_parabola test failed: {e}")
        import traceback
        traceback.print_exc()
    
    try:
        result = test_problem(
            'sinusoidal',
            load_sinusoidal_data,
            os.path.join(data_dir, 'sinusoidal_examples.pl'),
            os.path.join(data_dir, 'sinusoidal_BK.pl'),
            'sinusoidal'
        )
        if result:
            results.append(result)
    except Exception as e:
        print(f"Error: sinusoidal test failed: {e}")
        import traceback
        traceback.print_exc()
    
    try:
        result = test_problem(
            'crescent',
            load_crescent_data,
            os.path.join(data_dir, 'crescent_examples.pl'),
            os.path.join(data_dir, 'crescent_BK.pl'),
            'crescent'
        )
        if result:
            results.append(result)
    except Exception as e:
        print(f"Error: crescent test failed: {e}")
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
            print("\nAll tests passed!")
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
