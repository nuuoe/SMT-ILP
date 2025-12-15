#!/usr/bin/env python3
"""
Full comparison: PyGol+Z3 vs NumSynth across all datasets.

Runs multiple trials for standard error calculation and outputs results in table format:
- Table 1: Predictive accuracies (rounded to integers, with standard error)
- Table 2: Learning times (rounded to nearest second if >1s, with standard error)

Output saved to comparison_results.txt and comparison_results.json
"""

import sys
import os
import time
import json
import subprocess
import tempfile
import shutil
import re
import signal
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from contextlib import redirect_stdout, redirect_stderr
from io import StringIO

# Add paths
_current_file = os.path.abspath(__file__)
_numsynth_comparison_dir = os.path.dirname(_current_file)  # numsynth_comparison folder
_smt_ilp_dir = os.path.dirname(_numsynth_comparison_dir)  # SMT-ILP root
# NumSynth is in numsynth_comparison/numsynth-main/numsynth
_numsynth_dir = os.path.join(_numsynth_comparison_dir, 'numsynth-main', 'numsynth')
_numsynth_py = os.path.join(_numsynth_dir, 'popper.py')

sys.path.insert(0, _smt_ilp_dir)

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

# Configuration
NUM_TRIALS = 10  # Number of trials for standard error calculation (increased from 5 to 10)
VERBOSE = False  # Set to True for detailed output
RUN_SUBSET = True  # Set to True to run only geometry0 and geometry1 (NumSynth only works with these)
SUPPRESS_PYGOL_OUTPUT = True  # Suppress PyGol verbose output
# Per-problem timeout: Based on NUMSYNTH_FINDINGS.md:
# - halfplane3d: ~333s per trial × 10 trials = ~3330s (55.5 min)
# - conjunction/interval3d: >600s per trial (will timeout on first trial)
# Set to 4000s (~66 min) to allow 10 slow trials with buffer
NUMSYNTH_PER_PROBLEM_TIMEOUT = 4000  # ~66 minutes per problem (all trials combined)

# Problems to skip for NumSynth (known to timeout or fail)
NUMSYNTH_SKIP_PROBLEMS = {
    'geometry1_conjunction',  # Timeout >600s per trial
    'geometry1_interval3d',   # Timeout >600s per trial
}

# Import learners
try:
    from geometry0.iterative_pygol_z3_learner_geometry0 import IterativePyGolZ3Learner as Geo0Learner
    from geometry0.load_geometry0_data import load_interval_data, load_halfplane_data
except ImportError as e:
    if VERBOSE:
        print(f"Warning: Could not import geometry0: {e}")
    Geo0Learner = None

try:
    from geometry1.iterative_pygol_z3_learner_geometry1 import IterativePyGolZ3Learner as Geo1Learner
    from geometry1.load_geometry1_data import (
        load_halfplane3d_data, load_conjunction_data,
        load_multihalfplane_data, load_interval3d_data
    )
except ImportError as e:
    if VERBOSE:
        print(f"Warning: Could not import geometry1: {e}")
    Geo1Learner = None

try:
    from geometry2.iterative_pygol_z3_learner_geometry2 import IterativePyGolZ3Learner as Geo2Learner
    from geometry2.load_geometry2_data import (
        load_left_of_data, load_inside_data, load_touching_data,
        load_overlapping_data, load_between_data, load_aligned_data,
        load_closer_than_data, load_near_corner_data, load_adjacent_data,
        load_surrounds_data
    )
except ImportError as e:
    if VERBOSE:
        print(f"Warning: Could not import geometry2: {e}")
    Geo2Learner = None

try:
    from geometry3.iterative_pygol_z3_learner_geometry3 import IterativePyGolZ3Learner as Geo3Learner
    from geometry3.load_geometry3_data import (
        load_in_circle_data, load_in_ellipse_data, load_hyperbola_side_data,
        load_xy_less_than_data, load_quad_strip_data, load_union_halfplanes_data,
        load_circle_or_box_data, load_piecewise_data, load_fallback_region_data,
        load_donut_data, load_lshape_data, load_above_parabola_data,
        load_sinusoidal_data, load_crescent_data
    )
except ImportError as e:
    if VERBOSE:
        print(f"Warning: Could not import geometry3: {e}")
    Geo3Learner = None

try:
    from ip.iterative_pygol_z3_learner_ip_main import IterativePyGolZ3Learner as IPLearner
    from ip.load_ip_data import load_ip_data
except ImportError as e:
    if VERBOSE:
        print(f"Warning: Could not import IP: {e}")
    IPLearner = None


class NumSynthRunner:
    """Wrapper to run NumSynth on problems using the same pipeline as numsynth-main framework"""
    
    def __init__(self, timeout=600):
        self.timeout = timeout
        self.numsynth_dir = _numsynth_dir
        self.numsynth_py = _numsynth_py
        # Path to test.pl from numsynth-main framework (numsynth-main is in numsynth_comparison folder)
        self.test_pl = os.path.join(_numsynth_comparison_dir, 'numsynth-main', 'ilp-experiments', 'ilpexp', 'system', 'test.pl')
    
    def call_prolog(self, action: str, files_to_load: List[str], timeout: int = 600) -> str:
        """Call SWI-Prolog using the same approach as numsynth-main framework"""
        args = ["-g", action, "-t", "halt", "-q"]
        
        if len(files_to_load) == 1:
            args.append("-s")
            args.append(files_to_load[0])
        else:
            # Create temp file with load_files command
            with tempfile.NamedTemporaryFile(mode='w', suffix='.pl', delete=False) as temp_file:
                files = ', '.join([f'"{f}"' for f in files_to_load])
                cmd = f":- load_files([{files}],[silent(true)])."
                temp_file.write(cmd)
                temp_file.flush()
                temp_file_path = temp_file.name
            
            args.append("-s")
            args.append(temp_file_path)
        
        try:
            result = subprocess.run(
                ['swipl'] + args,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            if len(files_to_load) > 1:
                os.unlink(temp_file_path)
            return result.stdout.strip()
        except subprocess.TimeoutExpired:
            if len(files_to_load) > 1:
                try:
                    os.unlink(temp_file_path)
                except:
                    pass
            return ""
        except Exception as e:
            if len(files_to_load) > 1:
                try:
                    os.unlink(temp_file_path)
                except:
                    pass
            return ""
    
    def extract_solution_from_stats(self, stats_file: str) -> tuple:
        """Extract learned program and execution time from NumSynth stats.json file"""
        if not os.path.exists(stats_file):
            return None, None, None
        
        try:
            with open(stats_file, 'r') as f:
                stats = json.load(f)
            
            code = None
            exec_time = None
            
            # NumSynth stores solution in stats["solution"]["code"]
            if 'solution' in stats and stats['solution']:
                code = stats['solution'].get('code')
                if code:
                    exec_time = stats.get('final_exec_time', None)
                    return code, exec_time, stats
            
            # Also check best_programs
            if 'best_programs' in stats and stats['best_programs']:
                code = stats['best_programs'][-1].get('code')
                if code:
                    exec_time = stats.get('final_exec_time', None)
                    return code, exec_time, stats
            
            return None, None, stats
        except Exception:
            return None, None, None
    
    def evaluate_with_test_pl(self, solution: str, test_examples_file: str, bk_file: str) -> Optional[float]:
        """Evaluate learned program using test.pl (same as numsynth-main framework)"""
        if not solution or not os.path.exists(self.test_pl):
            return None
        
        # Create temporary solution file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.pl', delete=False) as solution_file:
            solution_file.write(solution)
            solution_file.flush()
            solution_file_path = solution_file.name
        
        try:
            # Use test.pl to evaluate (same as System.test() in numsynth-main)
            files_to_load = [self.test_pl, test_examples_file, bk_file, solution_file_path]
            result = self.call_prolog('print_conf_matrix.', files_to_load, timeout=600)
            
            # Parse confusion matrix: "TP,FN,TN,FP"
            if result and ',' in result:
                parts = result.strip().split(',')
                if len(parts) == 4:
                    try:
                        tp, fn, tn, fp = [int(x) for x in parts]
                        total = tp + fn + tn + fp
                        if total > 0:
                            accuracy = (tp + tn) / total
                            return accuracy
                    except ValueError:
                        pass
            
            return None
        finally:
            os.unlink(solution_file_path)
    
    def run(self, problem_dir: str, test_examples_file: str = None) -> Dict:
        """
        Run NumSynth on a problem directory.
        Returns dict with 'accuracy', 'time', 'success'
        """
        if not os.path.exists(self.numsynth_py):
            return {'success': False, 'error': 'NumSynth not found', 'accuracy': None, 'time': None}
        
        # Create temporary directory for this run
        temp_dir = tempfile.mkdtemp()
        stats_file = os.path.join(temp_dir, 'stats.json')
        
        try:
            # Copy problem files to temp directory
            exs_file = os.path.join(problem_dir, 'exs.pl')
            bk_file = os.path.join(problem_dir, 'bk.pl')
            bias_file = os.path.join(problem_dir, 'numsynth-bias.pl')
            
            for f, name in [(exs_file, 'exs.pl'), (bk_file, 'bk.pl'), (bias_file, 'numsynth-bias.pl')]:
                if not os.path.exists(f):
                    return {'success': False, 'error': f'Missing {name}', 'accuracy': None, 'time': None}
            
            shutil.copy(exs_file, os.path.join(temp_dir, 'exs.pl'))
            shutil.copy(bk_file, os.path.join(temp_dir, 'bk.pl'))
            shutil.copy(bias_file, os.path.join(temp_dir, 'bias.pl'))
            
            # Run NumSynth
            exs_path = os.path.join(temp_dir, 'exs.pl')
            bk_path = os.path.join(temp_dir, 'bk.pl')
            bias_path = os.path.join(temp_dir, 'bias.pl')
            
            cmd = [
                sys.executable, self.numsynth_py,
                '--ex-file', exs_path,
                '--bk-file', bk_path,
                '--bias-file', bias_path,
                '--stats-file', stats_file,
                '--numerical-reasoning'
            ]
            
            start_time = time.time()
            try:
                # NumSynth needs the numsynth directory in PYTHONPATH
                env = os.environ.copy()
                if 'PYTHONPATH' in env:
                    env['PYTHONPATH'] = f"{self.numsynth_dir}:{env['PYTHONPATH']}"
                else:
                    env['PYTHONPATH'] = self.numsynth_dir
                
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout,
                    cwd=self.numsynth_dir,
                    env=env
                )
                elapsed = time.time() - start_time
                
                if result.returncode != 0:
                    return {
                        'success': False,
                        'time': elapsed,
                        'accuracy': None,
                        'error': result.stderr[-200:] if result.stderr else 'Unknown error'
                    }
                
                if 'NO SOLUTION' in result.stdout:
                    return {
                        'success': False,
                        'time': elapsed,
                        'accuracy': None,
                        'error': 'No solution found'
                    }
                
                # Extract solution from stats.json
                solution, exec_time, stats = self.extract_solution_from_stats(stats_file)
                
                if not solution:
                    return {
                        'success': False,
                        'time': elapsed,
                        'accuracy': None,
                        'error': 'No solution in stats.json'
                    }
                
                # Evaluate on test examples (or training examples if no test file)
                eval_file = test_examples_file if test_examples_file and os.path.exists(test_examples_file) else exs_path
                accuracy = self.evaluate_with_test_pl(solution, eval_file, bk_path)
                
                return {
                    'success': True,
                    'time': exec_time if exec_time else elapsed,
                    'accuracy': accuracy if accuracy is not None else 0.0,
                    'program': solution
                }
            except subprocess.TimeoutExpired:
                return {
                    'success': False,
                    'time': self.timeout,
                    'accuracy': None,
                    'error': 'Timeout'
                }
        finally:
            shutil.rmtree(temp_dir)


def test_pygol_z3_single_trial(problem_config: Dict) -> Dict:
    """Run PyGol+Z3 on a problem for one trial"""
    try:
        dataset = problem_config['dataset']
        problem_name = problem_config['problem']
        load_func = problem_config['load_func']
        examples_file = problem_config['examples_file']
        bk_file = problem_config.get('bk_file')
        learner_class = problem_config['learner_class']
        learner_config = problem_config['learner_config']
        
        # Load data (IP uses task name, others use file paths)
        if dataset == 'ip':
            X, y, bk = load_func(examples_file)  # examples_file is task_name for IP
            if bk and 'original_bk_file' in learner_config['attributes']:
                learner_config['attributes']['original_bk_file'] = bk
        elif bk_file:
            X, y, _ = load_func(examples_file, bk_file)
        else:
            X, y, _ = load_func(examples_file)
        
        # Split train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=None, stratify=y
        )
        
        # Create and configure learner
        learner = learner_class(**learner_config['init_params'])
        for key, value in learner_config['attributes'].items():
            setattr(learner, key, value)
        
        # Train (suppress output if requested)
        start_time = time.time()
        if SUPPRESS_PYGOL_OUTPUT:
            # Redirect stdout/stderr to suppress PyGol verbose output
            with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
                learner.fit(X_train, y_train)
        else:
            learner.fit(X_train, y_train)
        elapsed = time.time() - start_time
        
        # Evaluate
        if hasattr(learner, 'learned_rules') and learner.learned_rules:
            y_pred = learner.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            return {
                'success': True,
                'accuracy': accuracy,
                'time': elapsed,
                'num_rules': len(learner.learned_rules)
            }
        else:
            return {
                'success': False,
                'accuracy': 0.0,
                'time': elapsed,
                'num_rules': 0
            }
    except Exception as e:
        if VERBOSE:
            print(f"Error in trial: {e}")
        return {
            'success': False,
            'accuracy': 0.0,
            'time': None,
            'num_rules': 0,
            'error': str(e)
        }


def test_numsynth_single_trial(problem_dir: str, test_examples_file: str = None) -> Dict:
    """Run NumSynth on a problem for one trial"""
    runner = NumSynthRunner(timeout=600)
    return runner.run(problem_dir, test_examples_file)


def run_comparison(problem_configs: List[Dict], numsynth_dirs: Dict[str, str]) -> Dict:
    """Run full comparison across all problems"""
    results = {}
    
    print("Running comparison across all datasets...")
    print(f"Number of trials per problem: {NUM_TRIALS}")
    print()
    
    for config in problem_configs:
        dataset = config['dataset']
        problem = config['problem']
        full_name = f"{dataset}_{problem}"
        
        print(f"Testing {full_name}...")
        
        # PyGol+Z3 trials
        pygol_accuracies = []
        pygol_times = []
        
        for trial in range(NUM_TRIALS):
            if VERBOSE:
                print(f"  PyGol+Z3 trial {trial+1}/{NUM_TRIALS}...")
            result = test_pygol_z3_single_trial(config)
            if result['success']:
                pygol_accuracies.append(result['accuracy'])
                if result['time']:
                    pygol_times.append(result['time'])
        
        # NumSynth trials
        numsynth_accuracies = []
        numsynth_times = []
        
        print(f"  Checking NumSynth for {full_name}...")
        print(f"    numsynth_dirs keys: {list(numsynth_dirs.keys())}")
        if full_name in numsynth_dirs:
            numsynth_dir = numsynth_dirs[full_name]
            
            # Skip NumSynth for problems that are known to timeout or fail
            if full_name in NUMSYNTH_SKIP_PROBLEMS:
                print(f"  [SKIP] NumSynth skipped for {full_name} (known to timeout/fail per NUMSYNTH_FINDINGS.md)")
            else:
                print(f"    Found NumSynth dir: {numsynth_dir}")
                # Check if NumSynth dataset exists
                if not os.path.exists(numsynth_dir):
                    print(f"  [WARNING] NumSynth dataset not found: {numsynth_dir}")
                else:
                    print(f"    NumSynth dataset exists")
                    # Create test examples file for NumSynth evaluation
                    # Note: We don't have X_test/y_test here since each trial splits differently
                    # NumSynth will evaluate on its own test set if provided, or use training examples
                    test_examples_file = None
                    
                    print(f"  Running NumSynth ({NUM_TRIALS} trials) with {NUMSYNTH_PER_PROBLEM_TIMEOUT}s timeout per problem...")
                    
                    # Set up timeout handler for per-problem timeout
                    timeout_occurred = False
                    
                    def timeout_handler(signum, frame):
                        nonlocal timeout_occurred
                        timeout_occurred = True
                        raise TimeoutError(f"NumSynth per-problem timeout ({NUMSYNTH_PER_PROBLEM_TIMEOUT}s) exceeded")
                    
                    # Set alarm for per-problem timeout (Unix only)
                    original_handler = None
                    if hasattr(signal, 'SIGALRM'):
                        original_handler = signal.signal(signal.SIGALRM, timeout_handler)
                        signal.alarm(NUMSYNTH_PER_PROBLEM_TIMEOUT)
                    
                    problem_start_time = time.time()
                    try:
                        for trial in range(NUM_TRIALS):
                            # Check if we've exceeded timeout (for non-Unix systems or as backup)
                            elapsed = time.time() - problem_start_time
                            if elapsed >= NUMSYNTH_PER_PROBLEM_TIMEOUT:
                                print(f"    Per-problem timeout ({NUMSYNTH_PER_PROBLEM_TIMEOUT}s) exceeded, stopping trials", flush=True)
                                timeout_occurred = True
                                break
                            
                            print(f"    NumSynth trial {trial+1}/{NUM_TRIALS}...", flush=True)
                            result = test_numsynth_single_trial(numsynth_dir, test_examples_file)
                            if result['success']:
                                print(f"      Trial {trial+1} SUCCESS: acc={result.get('accuracy', 'N/A'):.3f}, time={result.get('time', 0):.1f}s", flush=True)
                                if result['accuracy'] is not None:
                                    numsynth_accuracies.append(result['accuracy'])
                                if result['time']:
                                    numsynth_times.append(result['time'])
                            else:
                                print(f"      Trial {trial+1} FAILED: {result.get('error', 'Unknown error')}", flush=True)
                    except TimeoutError as e:
                        print(f"  NumSynth timed out after {NUMSYNTH_PER_PROBLEM_TIMEOUT}s: {str(e)}", flush=True)
                        timeout_occurred = True
                    finally:
                        # Cancel alarm
                        if hasattr(signal, 'SIGALRM') and original_handler is not None:
                            signal.alarm(0)
                            signal.signal(signal.SIGALRM, original_handler)
                    
                    # Report results
                    if timeout_occurred:
                        print(f"  NumSynth: TIMEOUT (stopped after {NUMSYNTH_PER_PROBLEM_TIMEOUT}s, {len(numsynth_accuracies)}/{NUM_TRIALS} trials completed)")
                    elif numsynth_accuracies:
                        mean_acc = np.mean(numsynth_accuracies)
                        std_acc = np.std(numsynth_accuracies, ddof=1) if len(numsynth_accuracies) > 1 else 0.0
                        se_acc = std_acc / np.sqrt(len(numsynth_accuracies)) if len(numsynth_accuracies) > 0 else 0.0
                        if not np.isnan(mean_acc) and not np.isnan(se_acc):
                            print(f"  NumSynth: {int(mean_acc*100)}% ± {int(se_acc*100)}% ({len(numsynth_accuracies)}/{NUM_TRIALS} successful)")
                        else:
                            print(f"  NumSynth: TIMEOUT (no valid results)")
                    else:
                        print(f"  NumSynth: All {NUM_TRIALS} trials failed")
                        # Always show error details when all trials fail
                        result = test_numsynth_single_trial(numsynth_dir, test_examples_file)
                        print(f"    Error: {result.get('error', 'Unknown error')}")
                    
                    # Clean up temp file
                    if test_examples_file and os.path.exists(test_examples_file):
                        os.unlink(test_examples_file)
        else:
            print(f"  [WARNING] NumSynth dataset not configured for {full_name}")
        
        # Calculate statistics
        pygol_acc_mean = np.mean(pygol_accuracies) if pygol_accuracies else 0.0
        pygol_acc_std = np.std(pygol_accuracies, ddof=1) if len(pygol_accuracies) > 1 else 0.0
        pygol_acc_se = pygol_acc_std / np.sqrt(len(pygol_accuracies)) if pygol_accuracies else 0.0
        
        pygol_time_mean = np.mean(pygol_times) if pygol_times else 0.0
        pygol_time_std = np.std(pygol_times, ddof=1) if len(pygol_times) > 1 else 0.0
        pygol_time_se = pygol_time_std / np.sqrt(len(pygol_times)) if pygol_times else 0.0
        
        numsynth_acc_mean = np.mean(numsynth_accuracies) if numsynth_accuracies else None
        numsynth_acc_std = np.std(numsynth_accuracies, ddof=1) if len(numsynth_accuracies) > 1 else 0.0
        numsynth_acc_se = numsynth_acc_std / np.sqrt(len(numsynth_accuracies)) if numsynth_accuracies and len(numsynth_accuracies) > 0 else None
        
        # Handle NaN values
        if numsynth_acc_mean is not None and np.isnan(numsynth_acc_mean):
            numsynth_acc_mean = None
        if numsynth_acc_se is not None and np.isnan(numsynth_acc_se):
            numsynth_acc_se = None
        
        numsynth_time_mean = np.mean(numsynth_times) if numsynth_times else None
        numsynth_time_std = np.std(numsynth_times, ddof=1) if len(numsynth_times) > 1 else 0.0
        numsynth_time_se = numsynth_time_std / np.sqrt(len(numsynth_times)) if numsynth_times and len(numsynth_times) > 0 else None
        
        # Handle NaN values for time
        if numsynth_time_mean is not None and np.isnan(numsynth_time_mean):
            numsynth_time_mean = None
        if numsynth_time_se is not None and np.isnan(numsynth_time_se):
            numsynth_time_se = None
        
        results[full_name] = {
            'pygol_z3': {
                'accuracy_mean': pygol_acc_mean,
                'accuracy_se': pygol_acc_se,
                'time_mean': pygol_time_mean,
                'time_se': pygol_time_se,
                'num_successful': len(pygol_accuracies)
            },
            'numsynth': {
                'accuracy_mean': numsynth_acc_mean,
                'accuracy_se': numsynth_acc_se,
                'time_mean': numsynth_time_mean,
                'time_se': numsynth_time_se,
                'num_successful': len(numsynth_accuracies) if numsynth_accuracies else 0
            }
        }
        
        print(f"  PyGol+Z3: {int(pygol_acc_mean*100)}% ± {int(pygol_acc_se*100)}%")
        if numsynth_acc_mean is not None and numsynth_acc_se is not None:
            print(f"  NumSynth: {int(numsynth_acc_mean*100)}% ± {int(numsynth_acc_se*100)}%")
        elif numsynth_acc_mean is not None:
            print(f"  NumSynth: {int(numsynth_acc_mean*100)}% ± N/A")
        else:
            print(f"  NumSynth: TIMEOUT/N/A")
        print()
    
    return results


def format_table(results: Dict) -> Tuple[str, str]:
    """Format results into Table 1 (accuracies) and Table 2 (times)"""
    
    # Table 1: Accuracies
    table1 = "Table 1: Predictive Accuracies\n"
    table1 += "=" * 80 + "\n"
    table1 += f"{'Task':<40} {'PyGol+Z3':<20} {'NumSynth':<20}\n"
    table1 += "-" * 80 + "\n"
    
    # Table 2: Times
    table2 = "Table 2: Learning Times (seconds)\n"
    table2 += "=" * 80 + "\n"
    table2 += f"{'Task':<40} {'PyGol+Z3':<20} {'NumSynth':<20}\n"
    table2 += "-" * 80 + "\n"
    
    for task_name in sorted(results.keys()):
        r = results[task_name]
        
        # Accuracy row
        pygol_acc = int(r['pygol_z3']['accuracy_mean'] * 100)
        pygol_acc_se = int(r['pygol_z3']['accuracy_se'] * 100)
        pygol_acc_str = f"{pygol_acc} ± {pygol_acc_se}"
        
        if r['numsynth']['accuracy_mean'] is not None:
            numsynth_acc = int(r['numsynth']['accuracy_mean'] * 100)
            numsynth_acc_se = int(r['numsynth']['accuracy_se'] * 100) if r['numsynth']['accuracy_se'] is not None else 0
            numsynth_acc_str = f"{numsynth_acc} ± {numsynth_acc_se}"
        else:
            numsynth_acc_str = "N/A"
        
        table1 += f"{task_name:<40} {pygol_acc_str:<20} {numsynth_acc_str:<20}\n"
        
        # Time row
        pygol_time = r['pygol_z3']['time_mean']
        if pygol_time >= 1.0:
            pygol_time_str = f"{int(round(pygol_time))} ± {int(round(r['pygol_z3']['time_se']))}"
        else:
            pygol_time_str = f"{pygol_time:.2f} ± {r['pygol_z3']['time_se']:.2f}"
        
        if r['numsynth']['time_mean'] is not None and r['numsynth']['time_se'] is not None:
            numsynth_time = r['numsynth']['time_mean']
            numsynth_time_se = r['numsynth']['time_se']
            if numsynth_time >= 1.0:
                numsynth_time_str = f"{int(round(numsynth_time))} ± {int(round(numsynth_time_se))}"
            else:
                numsynth_time_str = f"{numsynth_time:.2f} ± {numsynth_time_se:.2f}"
        else:
            numsynth_time_str = "N/A"
        
        table2 += f"{task_name:<40} {pygol_time_str:<20} {numsynth_time_str:<20}\n"
    
    return table1, table2


def format_markdown(results: Dict) -> str:
    """Format results as Markdown tables - one table per dataset with all metrics"""
    md = "# PyGol+Z3 vs NumSynth Comparison Results\n\n"
    md += f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    md += f"Results from {NUM_TRIALS} trial(s) per problem.\n\n"
    
    # Group results by dataset
    datasets = {}
    for task_name in sorted(results.keys()):
        # Extract dataset name (e.g., "geometry0_interval" -> "geometry0")
        parts = task_name.split('_', 1)
        dataset = parts[0] if len(parts) > 1 else 'other'
        problem = parts[1] if len(parts) > 1 else task_name
        
        if dataset not in datasets:
            datasets[dataset] = []
        datasets[dataset].append((task_name, problem, results[task_name]))
    
    # Create one table per dataset
    for dataset in sorted(datasets.keys()):
        md += f"## {dataset.upper()}\n\n"
        md += "| Task | PyGol+Z3 Acc (%) | NumSynth Acc (%) | PyGol+Z3 Time (s) | NumSynth Time (s) |\n"
        md += "|------|------------------|------------------|-------------------|-------------------|\n"
        
        for task_name, problem, r in datasets[dataset]:
            # Accuracy values
            pygol_acc = int(r['pygol_z3']['accuracy_mean'] * 100)
            pygol_acc_se = int(r['pygol_z3']['accuracy_se'] * 100)
            pygol_acc_str = f"{pygol_acc} ± {pygol_acc_se}"
            
            if r['numsynth']['accuracy_mean'] is not None:
                numsynth_acc = int(r['numsynth']['accuracy_mean'] * 100)
                numsynth_acc_se = int(r['numsynth']['accuracy_se'] * 100) if r['numsynth']['accuracy_se'] is not None else 0
                numsynth_acc_str = f"{numsynth_acc} ± {numsynth_acc_se}"
            else:
                numsynth_acc_str = "N/A"
            
            # Time values
            pygol_time = r['pygol_z3']['time_mean']
            if pygol_time >= 1.0:
                pygol_time_str = f"{int(round(pygol_time))} ± {int(round(r['pygol_z3']['time_se']))}"
            else:
                pygol_time_str = f"{pygol_time:.2f} ± {r['pygol_z3']['time_se']:.2f}"
            
            if r['numsynth']['time_mean'] is not None and r['numsynth']['time_se'] is not None:
                numsynth_time = r['numsynth']['time_mean']
                numsynth_time_se = r['numsynth']['time_se']
                if numsynth_time >= 1.0:
                    numsynth_time_str = f"{int(round(numsynth_time))} ± {int(round(numsynth_time_se))}"
                else:
                    numsynth_time_str = f"{numsynth_time:.2f} ± {numsynth_time_se:.2f}"
            else:
                numsynth_time_str = "N/A"
            
            md += f"| {problem} | {pygol_acc_str} | {numsynth_acc_str} | {pygol_time_str} | {numsynth_time_str} |\n"
        
        md += "\n"
    
    # Summary statistics
    md += "## Summary\n\n"
    
    # Calculate averages per dataset
    for dataset in sorted(datasets.keys()):
        dataset_results = [r for _, _, r in datasets[dataset]]
        pygol_accs = [r['pygol_z3']['accuracy_mean'] for r in dataset_results if r['pygol_z3']['accuracy_mean'] > 0]
        numsynth_accs = [r['numsynth']['accuracy_mean'] for r in dataset_results if r['numsynth']['accuracy_mean'] is not None]
        
        if pygol_accs:
            avg_pygol = np.mean(pygol_accs) * 100
            md += f"- **{dataset.upper()} - PyGol+Z3 Average Accuracy:** {avg_pygol:.2f}%\n"
        
        if numsynth_accs:
            avg_numsynth = np.mean(numsynth_accs) * 100
            md += f"- **{dataset.upper()} - NumSynth Average Accuracy:** {avg_numsynth:.2f}%\n"
    
    # Overall averages
    all_pygol_accs = [r['pygol_z3']['accuracy_mean'] for r in results.values() if r['pygol_z3']['accuracy_mean'] > 0]
    all_numsynth_accs = [r['numsynth']['accuracy_mean'] for r in results.values() if r['numsynth']['accuracy_mean'] is not None]
    
    if all_pygol_accs:
        avg_pygol = np.mean(all_pygol_accs) * 100
        md += f"\n- **Overall PyGol+Z3 Average Accuracy:** {avg_pygol:.2f}%\n"
    
    if all_numsynth_accs:
        avg_numsynth = np.mean(all_numsynth_accs) * 100
        md += f"- **Overall NumSynth Average Accuracy:** {avg_numsynth:.2f}%\n"
    
    return md


def main():
    """Main comparison function"""
    
    # Define all problem configurations
    problem_configs = []
    numsynth_dirs = {}
    
    # Geometry0
    if Geo0Learner:
        geo0_dir = os.path.join(_experiments_dir, 'geometry0', 'data')
        # numsynth_datasets is in numsynth_comparison folder, not experiments folder
        numsynth_base = os.path.join(_numsynth_comparison_dir, 'numsynth_datasets', 'geometry0')
        
        # Interval
        interval_ex = os.path.join(geo0_dir, 'interval_examples.pl')
        if os.path.exists(interval_ex):
            problem_configs.append({
                'dataset': 'geometry0',
                'problem': 'interval',
                'load_func': load_interval_data,
                'examples_file': interval_ex,
                'bk_file': None,
                'learner_class': Geo0Learner,
                'learner_config': {
                    'init_params': {
                        'max_iterations': 5,
                        'max_literals': 3,
                        'verbose': False,
                        'pygol_timeout': 45
                    },
                    'attributes': {
                        'dataset_type': 'geometry',
                        'target_predicate': 'interval',
                        'original_bk_file': None,
                        'dataset_config': {
                            'learning_strategy': 'arithmetic',
                            'arithmetic_bounds': (-100, 100),
                            'use_arithmetic_learning': True,
                            'use_optimization': True,  # Enable Z3 for numerical optimization
                        }
                    }
                }
            })
            numsynth_dirs['geometry0_interval'] = os.path.join(numsynth_base, 'interval')
        
        # Halfplane
        halfplane_ex = os.path.join(geo0_dir, 'halfplane_examples.pl')
        if os.path.exists(halfplane_ex):
            problem_configs.append({
                'dataset': 'geometry0',
                'problem': 'halfplane',
                'load_func': load_halfplane_data,
                'examples_file': halfplane_ex,
                'bk_file': None,
                'learner_class': Geo0Learner,
                'learner_config': {
                    'init_params': {
                        'max_iterations': 5,
                        'max_literals': 3,
                        'verbose': False,
                        'pygol_timeout': 45
                    },
                    'attributes': {
                        'dataset_type': 'geometry',
                        'target_predicate': 'halfplane',
                        'original_bk_file': None,
                        'dataset_config': {
                            'learning_strategy': 'arithmetic',
                            'arithmetic_bounds': (-100, 100),
                            'use_arithmetic_learning': True,
                            'use_optimization': True,  # Enable Z3 for numerical optimization
                        }
                    }
                }
            })
            numsynth_dirs['geometry0_halfplane'] = os.path.join(numsynth_base, 'halfplane')
    
    # Geometry1
    if Geo1Learner:
        geo1_dir = os.path.join(_experiments_dir, 'geometry1', 'data')
        # numsynth_datasets is in numsynth_comparison folder, not experiments folder
        numsynth_base = os.path.join(_numsynth_comparison_dir, 'numsynth_datasets', 'geometry1')
        
        problems = [
            ('halfplane3d', load_halfplane3d_data),
            ('conjunction', load_conjunction_data),
            ('multihalfplane', load_multihalfplane_data),
            ('interval3d', load_interval3d_data),
        ]
        
        for prob_name, load_func in problems:
            ex_file = os.path.join(geo1_dir, f'{prob_name}_examples.pl')
            bk_file = os.path.join(geo1_dir, f'{prob_name}_BK.pl')
            if os.path.exists(ex_file) and os.path.exists(bk_file):
                problem_configs.append({
                    'dataset': 'geometry1',
                    'problem': prob_name,
                    'load_func': load_func,
                    'examples_file': ex_file,
                    'bk_file': bk_file,
                    'learner_class': Geo1Learner,
                    'learner_config': {
                        'init_params': {
                            'max_iterations': 10,
                            'max_literals': 6,
                            'verbose': False,
                            'pygol_timeout': 120
                        },
                        'attributes': {
                            'dataset_type': 'geometry',
                            'target_predicate': prob_name.lower(),
                            'original_bk_file': bk_file,
                            'dataset_config': {
                                'learning_strategy': 'arithmetic',
                                'arithmetic_bounds': (-100, 100),
                                'use_arithmetic_learning': True,
                                'use_optimization': True,  # Enable Z3 for numerical optimization
                            }
                        }
                    }
                })
                numsynth_dirs[f'geometry1_{prob_name}'] = os.path.join(numsynth_base, prob_name)
    
    # Geometry2 - using individual loaders (skip if running subset)
    if Geo2Learner and not RUN_SUBSET:
        geo2_dir = os.path.join(_experiments_dir, 'geometry2', 'data')
        # numsynth_datasets is in numsynth_comparison folder, not experiments folder
        numsynth_base = os.path.join(_numsynth_comparison_dir, 'numsynth_datasets', 'geometry2')
        
        problems = [
            ('left_of', load_left_of_data),
            ('inside', load_inside_data),
            ('touching', load_touching_data),
            ('overlapping', load_overlapping_data),
            ('between', load_between_data),
            ('aligned', load_aligned_data),
            ('closer_than', load_closer_than_data),
            ('near_corner', load_near_corner_data),
            ('adjacent', load_adjacent_data),
            ('surrounds', load_surrounds_data),
        ]
        
        for prob_name, load_func in problems:
            ex_file = os.path.join(geo2_dir, f'{prob_name}_examples.pl')
            bk_file = os.path.join(geo2_dir, f'{prob_name}_BK.pl')
            if os.path.exists(ex_file) and os.path.exists(bk_file):
                problem_configs.append({
                    'dataset': 'geometry2',
                    'problem': prob_name,
                    'load_func': load_func,
                    'examples_file': ex_file,
                    'bk_file': bk_file,
                    'learner_class': Geo2Learner,
                    'learner_config': {
                        'init_params': {
                            'max_iterations': 3,  # Good settings: 3 iterations (was 10)
                            'max_literals': 5,    # Good settings: 5 literals (was 4)
                            'verbose': False,
                            'pygol_timeout': 30   # Good settings: 30s timeout (was 120s)
                        },
                        'attributes': {
                            'dataset_type': 'geometry',  # Fixed: was 'geometry2', should be 'geometry'
                            'target_predicate': prob_name,
                            'original_bk_file': bk_file,
                            'dataset_config': {  # Added: required for geometry2 problems
                                'learning_strategy': 'arithmetic',
                                'arithmetic_bounds': (-100, 100),
                                'use_arithmetic_learning': True,
                                'use_distance_learning': False,
                                'use_optimization': True,  # Enable Z3 for numerical optimization
                            }
                        }
                    }
                })
                numsynth_dirs[f'geometry2_{prob_name}'] = os.path.join(numsynth_base, prob_name)
    
    # Geometry3 - using individual loaders (skip if running subset)
    if Geo3Learner and not RUN_SUBSET:
        geo3_dir = os.path.join(_experiments_dir, 'geometry3', 'data')
        # numsynth_datasets is in numsynth_comparison folder, not experiments folder
        numsynth_base = os.path.join(_numsynth_comparison_dir, 'numsynth_datasets', 'geometry3')
        
        problems = [
            ('in_circle', load_in_circle_data),
            ('in_ellipse', load_in_ellipse_data),
            ('hyperbola_side', load_hyperbola_side_data),
            ('xy_less_than', load_xy_less_than_data),
            ('quad_strip', load_quad_strip_data),
            ('union_halfplanes', load_union_halfplanes_data),
            ('circle_or_box', load_circle_or_box_data),
            ('piecewise', load_piecewise_data),
            ('fallback_region', load_fallback_region_data),
            ('donut', load_donut_data),
            ('lshape', load_lshape_data),
            ('above_parabola', load_above_parabola_data),
            ('sinusoidal', load_sinusoidal_data),
            ('crescent', load_crescent_data),
        ]
        
        for prob_name, load_func in problems:
            ex_file = os.path.join(geo3_dir, f'{prob_name}_examples.pl')
            bk_file = os.path.join(geo3_dir, f'{prob_name}_BK.pl')
            if os.path.exists(ex_file) and os.path.exists(bk_file):
                problem_configs.append({
                    'dataset': 'geometry3',
                    'problem': prob_name,
                    'load_func': load_func,
                    'examples_file': ex_file,
                    'bk_file': bk_file,
                    'learner_class': Geo3Learner,
                    'learner_config': {
                        'init_params': {
                            'max_iterations': 10,
                            'max_literals': 6,
                            'verbose': False,
                            'pygol_timeout': 120
                        },
                        'attributes': {
                            'dataset_type': 'geometry3',
                            'target_predicate': prob_name,
                            'original_bk_file': bk_file,
                            'dataset_config': {
                                'learning_strategy': 'arithmetic',
                                'arithmetic_bounds': (-100, 100),
                                'use_arithmetic_learning': True,
                                'use_optimization': True,  # Enable Z3 for nonlinear/disjunctive constraints
                            }
                        }
                    }
                })
                numsynth_dirs[f'geometry3_{prob_name}'] = os.path.join(numsynth_base, prob_name)
    
    # IP (skip if running subset)
    if IPLearner and not RUN_SUBSET:
        # numsynth_datasets is in numsynth_comparison folder, not experiments folder
        numsynth_base = os.path.join(_numsynth_comparison_dir, 'numsynth_datasets', 'ip')
        
        ip_tasks = ['ip1_active', 'ip2_active', 'ip3_active', 'ip3_threshold', 'ip4_high_score']
        
        for task_name in ip_tasks:
            problem_configs.append({
                'dataset': 'ip',
                'problem': task_name,
                'load_func': load_ip_data,
                'examples_file': task_name,  # load_ip_data takes task_name directly
                'bk_file': None,
                'learner_class': IPLearner,
                'learner_config': {
                    'init_params': {
                        'max_iterations': 1,
                        'max_literals': 4,
                        'verbose': False,
                        'pygol_timeout': 180
                    },
                    'attributes': {
                        'dataset_type': 'ip',
                        'target_predicate': task_name,
                        'original_bk_file': None,  # Will be set by load_ip_data
                        'dataset_config': {
                            'empty_constant_set': True,
                            'enable_predicate_invention': True,
                            'use_optimization': True
                        }
                    }
                }
            })
            numsynth_dirs[f'ip_{task_name}'] = os.path.join(numsynth_base, task_name)
    
    print(f"Found {len(problem_configs)} problems to test")
    if RUN_SUBSET:
        print("Running SUBSET: geometry0 and geometry1 only")
    else:
        print("Running ALL datasets")
    print(f"Number of trials per problem: {NUM_TRIALS}")
    print()
    
    # Run comparison
    results = run_comparison(problem_configs, numsynth_dirs)
    
    # Format and save results
    table1, table2 = format_table(results)
    md_content = format_markdown(results)
    
    output_text = table1 + "\n\n" + table2
    output_file = os.path.join(_numsynth_comparison_dir, 'comparison_results.txt')
    with open(output_file, 'w') as f:
        f.write(output_text)
    
    json_file = os.path.join(_numsynth_comparison_dir, 'comparison_results.json')
    with open(json_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    md_file = os.path.join(_numsynth_comparison_dir, 'comparison_results.md')
    with open(md_file, 'w') as f:
        f.write(md_content)
    
    print("COMPARISON COMPLETE")
    print()
    print(output_text)
    print()
    print(f"Results saved to:")
    print(f"  - {output_file}")
    print(f"  - {json_file}")
    print(f"  - {md_file}")


if __name__ == "__main__":
    main()

