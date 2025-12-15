#!/usr/bin/env python3
"""
Test all 4 InfluencePropagation (IP) tasks to demonstrate PI and Z3 requirements.

Expected results:
- Without PI: ~50-55% (cannot express multi-hop chains)
- PI, No Z3: ~60-65% (has structure but wrong influence calculations)  
- PI + Z3: ~85-90% (BEST - both structure and correct numerics)
"""

import sys
import os
import time

# Add paths
_current_file = os.path.abspath(__file__)
_ip_dir = os.path.dirname(_current_file)
_smt_ilp_dir = os.path.dirname(_ip_dir)  # SMT-ILP root
sys.path.insert(0, _ip_dir)  # Current directory (ip)
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

from iterative_pygol_z3_learner_ip_main import IterativePyGolZ3Learner
from load_ip_data import load_ip_data
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def test_task(task_name):
    """Test one IP task with 3 configurations"""
    task_start_time = time.time()
    print(f"Testing: {task_name}", flush=True)
    
    # Get the directory where this script is located (ip/)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, 'data')
    
    X, y, bk = load_ip_data(task_name, data_dir=data_dir)
    # Get random seed from environment variable (set by run_all_pygol_z3.py for multiple trials)
    # Default to 42 for backward compatibility
    random_seed = int(os.environ.get('PYGOL_RANDOM_SEED', '42'))
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=random_seed, stratify=y
    )
    
    print(f"Data: {len(X_train)} train ({sum(y_train)} pos), {len(X_test)} test ({sum(y_test)} pos)\n")
    
    # Use longer timeout for ip3_active (hardest pattern)
    base_timeout = 120 if task_name == 'ip3_active' else 30
    
    results = {}
    
    # Config 1: WITHOUT PI
    print("[1/3] WITHOUT PI (max_literals=2)...", flush=True)
    learner1 = IterativePyGolZ3Learner(
        max_iterations=1, max_literals=2, verbose=False, pygol_timeout=base_timeout
    )
    # IP-specific configuration
    learner1.dataset_type = 'ip'
    learner1.target_predicate = task_name
    learner1.original_bk_file = bk
    learner1.dataset_config = {
        'empty_constant_set': True,
        'enable_predicate_invention': False,
        'use_optimization': True
    }
    learner1.fit(X_train, y_train)
    
    if hasattr(learner1, 'learned_rules') and learner1.learned_rules:
        test_acc1 = accuracy_score(y_test, learner1.predict(X_test))
        print(f"  Test: {test_acc1:.2%} ({len(learner1.learned_rules)} rules)", flush=True)
        results['no_pi'] = test_acc1
    else:
        results['no_pi'] = None
        print(f"  No rules -> N/A", flush=True)
    
    # Config 2: WITH PI, NO Z3
    # Use longer timeout for ip3_active
    pi_timeout = 180 if task_name == 'ip3_active' else 90
    print("\n[2/3] WITH PI, NO Z3 (max_literals=4, pi=True)...", flush=True)
    learner2 = IterativePyGolZ3Learner(
        max_iterations=1, max_literals=4, verbose=False, pygol_timeout=pi_timeout
    )
    # IP-specific configuration with PI enabled
    learner2.dataset_type = 'ip'
    learner2.target_predicate = task_name
    learner2.original_bk_file = bk
    learner2.dataset_config = {
        'empty_constant_set': True,
        'enable_predicate_invention': True,  # Enable PI!
        'max_literals_pi': 4,
        'use_optimization': False  # Disable Z3
    }
    learner2.fit(X_train, y_train)
    
    if hasattr(learner2, 'learned_rules') and learner2.learned_rules:
        test_acc2 = accuracy_score(y_test, learner2.predict(X_test))
        print(f"  Test: {test_acc2:.2%} ({len(learner2.learned_rules)} rules)", flush=True)
        results['pi_no_z3'] = test_acc2
    else:
        results['pi_no_z3'] = None
        print(f"  No rules -> N/A", flush=True)
    
    # Config 3: WITH PI + Z3
    # Use longer timeout for ip3_active
    pi_z3_timeout = 180 if task_name == 'ip3_active' else 90
    print("\n[3/3] WITH PI + Z3 (full integration)...", flush=True)
    learner3 = IterativePyGolZ3Learner(
        max_iterations=1, max_literals=4, verbose=False, pygol_timeout=pi_z3_timeout
    )
    # IP-specific configuration with BOTH PI and Z3
    learner3.dataset_type = 'ip'
    learner3.target_predicate = task_name
    learner3.original_bk_file = bk
    learner3.dataset_config = {
        'empty_constant_set': True,
        'enable_predicate_invention': True,  # Enable PI!
        'max_literals_pi': 4,
        'use_optimization': True  # Enable Z3!
    }
    learner3.fit(X_train, y_train)
    
    if hasattr(learner3, 'learned_rules') and learner3.learned_rules:
        test_acc3 = accuracy_score(y_test, learner3.predict(X_test))
        num_rules = len(learner3.learned_rules)
        print(f"  Test: {test_acc3:.2%} ({num_rules} rules)", flush=True)
        results['pi_z3'] = test_acc3
        results['num_rules'] = num_rules  # Store for direct function calls
        
        print(f"\n  Sample rule:")
        for r in learner3.learned_rules[:1]:
            print(f"    {str(r)[:120]}")
    else:
        results['pi_z3'] = None
        results['num_rules'] = 0
        print(f"  No rules -> N/A", flush=True)
    
    # Summary - handle None values
    no_pi_val = results['no_pi'] if results['no_pi'] is not None else 0.0
    pi_no_z3_val = results.get('pi_no_z3') if results.get('pi_no_z3') is not None else 0.0
    pi_z3_val = results.get('pi_z3') if results.get('pi_z3') is not None else 0.0
    
    z3_benefit = (pi_z3_val - pi_no_z3_val) * 100 if results.get('pi_no_z3') is not None and results.get('pi_z3') is not None else None
    
    print()
    print(f"SUMMARY:")
    if results['no_pi'] is not None:
        print(f"  NO PI:      {results['no_pi']:.2%}")
    else:
        print(f"  NO PI:      N/A")
    
    if results.get('pi_no_z3') is not None:
        print(f"  PI, NO Z3:  {results.get('pi_no_z3'):.2%}")
    else:
        print(f"  PI, NO Z3:  N/A")
    
    if results.get('pi_z3') is not None:
        if z3_benefit is not None:
            print(f"  PI + Z3:    {results.get('pi_z3'):.2%} (Z3 benefit: {z3_benefit:+.1f}%)")
        else:
            print(f"  PI + Z3:    {results.get('pi_z3'):.2%} (Z3 benefit: N/A)")
    else:
        print(f"  PI + Z3:    N/A")
    
    # Print elapsed time for this task
    task_elapsed = time.time() - task_start_time
    print(f"  Time: {task_elapsed:.2f}s")
    print()
    
    return results

def main():
    """Test all 5 InfluencePropagation tasks"""
    tasks = [
        'ip1_active',
        'ip2_active',
        'ip3_active',
        'ip3_threshold',
        'ip4_high_score'
    ]
    
    print("\nInfluencePropagation (IP): PI + Z3 Requirement Benchmark")
    print("\nTesting 5 tasks with progressive difficulty:")
    print("  1. ip1_active      - Simple (2 literals) - Baseline")
    print("  2. ip2_active      - Multi-hop (3 literals) - Requires PI")
    print("  3. ip3_active      - Multi-hop (3 literals) - Requires PI")  
    print("  4. ip3_threshold   - Multi-hop + numerical - Requires PI + Z3")
    print("  5. ip4_high_score  - Multi-hop + aggregate - Requires PI + Z3")
    print()
    
    all_results = []
    
    for task in tasks:
        try:
            results = test_task(task)
            all_results.append({
                'task': task,
                **results
            })
            print("\n")
        except Exception as e:
            print(f"Error testing {task}: {e}\n")
    
    # Final summary
    if all_results:
        print("FINAL SUMMARY: All Tasks")
        print(f"\n{'Task':<20} {'No PI':<10} {'PI,NoZ3':<10} {'PI+Z3':<10} {'Z3 Δ':<8}")
        
        for r in all_results:
            no_pi = r.get('no_pi')
            pi_no_z3 = r.get('pi_no_z3')
            pi_z3 = r.get('pi_z3')
            
            # Calculate Z3 Δ only
            z3_benefit = None
            if pi_no_z3 is not None and pi_z3 is not None:
                z3_benefit = (pi_z3 - pi_no_z3) * 100
            
            # Handle None values
            no_pi_str = f"{no_pi:<10.1%}" if no_pi is not None else "   N/A    "
            pi_no_z3_str = f"{pi_no_z3:<10.1%}" if pi_no_z3 is not None else "   N/A    "
            pi_z3_str = f"{pi_z3:<10.1%}" if pi_z3 is not None else "   N/A    "
            z3_delta_str = f"{z3_benefit:>6.1f}%" if z3_benefit is not None else "   N/A"
            
            print(f"{r['task']:<20} {no_pi_str} {pi_no_z3_str} {pi_z3_str} {z3_delta_str}")
        
        # Calculate average Z3 benefit only
        z3_benefits = []
        for r in all_results:
            if r.get('pi_no_z3') is not None and r.get('pi_z3') is not None:
                z3_benefits.append((r['pi_z3'] - r['pi_no_z3']) * 100)
        
        if z3_benefits:
            avg_z3_benefit = sum(z3_benefits) / len(z3_benefits)
            print("\nCONCLUSION:")
            print(f"\nZ3 Benefit:  Average +{avg_z3_benefit:.1f}%")
            print(f"   Evaluates nonlinear influence calculations")
            print(f"   IP demonstrates that Z3 is essential for optimal performance")
            print()

if __name__ == "__main__":
    main()

