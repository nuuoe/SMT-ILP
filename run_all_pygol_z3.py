#!/usr/bin/env python3
"""
This script calls the existing test_all.py scripts for each dataset
and aggregates their results into a unified report format.
"""

import sys
import os
import re
import subprocess
import json
import time
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

# Add paths
_current_file = os.path.abspath(__file__)
_smt_ilp_dir = os.path.dirname(_current_file)  # SMT-ILP root
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
NUM_TRIALS = 5  # Number of trials per problem (for statistical significance)
VERBOSE = False
SUPPRESS_PYGOL_OUTPUT = True  # Suppress PyGol's verbose output when calling test scripts


def parse_ip_output(output: str, dataset: str) -> List[Dict]:
    """
    Parse IP test output which has a different format.
    Extracts No PI, PI+NoZ3, PI+Z3 test accuracies and calculates Z3 Δ.
    
    Uses two methods:
    1. Parse SUMMARY section (most reliable)
    2. Fall back to pattern matching if SUMMARY not found
    """
    results = []
    lines = output.split('\n')
    
    # Method 1: Parse SUMMARY sections (most reliable)
    # Each task ends with a SUMMARY that has all three values
    current_task = None
    task_summaries = {}  # task_name -> {no_pi, pi_no_z3, pi_z3, num_rules}
    
    i = 0
    while i < len(lines):
        line = lines[i]
        
        # Find task name: "Testing: ip1_active"
        if line.startswith('Testing:'):
            current_task = line.replace('Testing:', '').strip()
            task_summaries[current_task] = {'no_pi': None, 'pi_no_z3': None, 'pi_z3': None, 'num_rules': 0, 'time': None}
        
        # Parse SUMMARY section for current task
        elif line.strip() == 'SUMMARY:' and current_task:
            # Look ahead for the three values
            for j in range(i+1, min(i+15, len(lines))):  # Increased to 15 to catch Time line
                summary_line = lines[j].strip()
                # NO PI: X.XX% or N/A
                if summary_line.startswith('NO PI:'):
                    match = re.search(r'NO PI:\s*([\d.]+)%', summary_line)
                    if match:
                        task_summaries[current_task]['no_pi'] = float(match.group(1)) / 100.0
                    elif 'N/A' in summary_line:
                        task_summaries[current_task]['no_pi'] = None
                # PI, NO Z3: X.XX% or N/A
                elif summary_line.startswith('PI, NO Z3:') or summary_line.startswith('PI,NOZ3:'):
                    match = re.search(r'PI[,\s]+NO Z3:\s*([\d.]+)%', summary_line)
                    if match:
                        task_summaries[current_task]['pi_no_z3'] = float(match.group(1)) / 100.0
                    elif 'N/A' in summary_line:
                        task_summaries[current_task]['pi_no_z3'] = None
                # PI + Z3: X.XX% or N/A
                elif summary_line.startswith('PI + Z3:') or summary_line.startswith('PI+Z3:'):
                    match = re.search(r'PI\s*\+\s*Z3:\s*([\d.]+)%', summary_line)
                    if match:
                        task_summaries[current_task]['pi_z3'] = float(match.group(1)) / 100.0
                    elif 'N/A' in summary_line:
                        task_summaries[current_task]['pi_z3'] = None
                # Time: X.XXs
                elif summary_line.startswith('Time:'):
                    match = re.search(r'Time:\s*([\d.]+)s', summary_line)
                    if match:
                        task_summaries[current_task]['time'] = float(match.group(1))
                    # Stop after finding Time (last line after SUMMARY)
                    break
        
        i += 1
    
    # Method 2: Fall back to pattern matching for any tasks not found in SUMMARY
    # Also use pattern matching to extract num_rules
    current_task = None
    task_data = {}
    
    for i, line in enumerate(lines):
        # Find task name: "Testing: ip1_active"
        if line.startswith('Testing:'):
            # Save previous task
            if current_task and current_task not in task_summaries:
                # Only use pattern matching if SUMMARY didn't work
                z3_delta = None
                if task_data.get('pi_z3') is not None and task_data.get('pi_no_z3') is not None:
                    z3_delta = (task_data['pi_z3'] - task_data['pi_no_z3']) * 100
                
                pi_z3_val = task_data.get('pi_z3')
                results.append({
                    'dataset': dataset,
                    'problem': current_task,
                    'num_rules': task_data.get('num_rules', 0),
                    'train_accuracy': None,
                    'test_accuracy': pi_z3_val,
                    'test_f1': None,
                    'no_pi': task_data.get('no_pi'),
                    'pi_no_z3': task_data.get('pi_no_z3'),
                    'pi_z3': pi_z3_val,
                    'z3_delta': z3_delta,
                    'status': 'PASS' if (pi_z3_val is not None and pi_z3_val > 0.5) else 'FAIL',
                    'passed': pi_z3_val is not None and pi_z3_val > 0.5
                })
            
            # Start new task
            current_task = line.replace('Testing:', '').strip()
            task_data = {}
            # Initialize from SUMMARY if available
            if current_task in task_summaries:
                task_data = task_summaries[current_task].copy()
        
        # Extract num_rules from Test: lines (SUMMARY doesn't have this)
        # Look for "Test: X.XX% (N rules)" pattern, prefer PI+Z3 (most reliable)
        if 'Test:' in line and 'rules' in line and current_task:
            match = re.search(r'Test:\s*[\d.]+\%\s*\((\d+)\s*rules\)', line)
            if match:
                num_rules_val = int(match.group(1))
                # If this is after [3/3] or PI+Z3, it's the most reliable
                if '[3/3]' in lines[max(0, i-5):i] or 'PI + Z3' in lines[max(0, i-5):i]:
                    task_data['num_rules'] = num_rules_val
                    # Also update task_summaries if this task is there
                    if current_task in task_summaries:
                        task_summaries[current_task]['num_rules'] = num_rules_val
                elif 'num_rules' not in task_data or task_data['num_rules'] == 0:
                    task_data['num_rules'] = num_rules_val
                    # Also update task_summaries if this task is there
                    if current_task in task_summaries:
                        task_summaries[current_task]['num_rules'] = num_rules_val
        
        # Pattern matching fallback (only if SUMMARY didn't work)
        if current_task and current_task not in task_summaries:
            # Extract No PI test accuracy: "[1/3] WITHOUT PI..."
            if '[1/3]' in line or ('WITHOUT' in line and 'PI' in line):
                for j in range(i+1, min(i+100, len(lines))):  # Increased search window
                    next_line = lines[j].strip()
                    # Skip progress bar lines (contain | and % but not Test:)
                    if '|' in next_line and '%' in next_line and 'Test:' not in next_line:
                        continue
                    if 'Test:' in next_line and 'rules' in next_line:
                        match = re.search(r'Test:\s*([\d.]+)%\s*\((\d+)\s*rules\)', next_line)
                        if match:
                            test_acc = float(match.group(1)) / 100.0
                            task_data['no_pi'] = test_acc
                            break
                    elif 'No rules' in next_line or ('N/A' in next_line and 'Test:' not in next_line):
                        task_data['no_pi'] = None
                        break
            
            # Extract PI+NoZ3 test accuracy: "[2/3] WITH PI, NO Z3..."
            if '[2/3]' in line or ('WITH PI' in line and 'NO Z3' in line) or 'PI, NO Z3' in line.upper():
                for j in range(i+1, min(i+100, len(lines))):  # Increased search window
                    next_line = lines[j].strip()
                    # Skip progress bar lines
                    if '|' in next_line and '%' in next_line and 'Test:' not in next_line:
                        continue
                    if 'Test:' in next_line and 'rules' in next_line:
                        match = re.search(r'Test:\s*([\d.]+)%\s*\((\d+)\s*rules\)', next_line)
                        if match:
                            test_acc = float(match.group(1)) / 100.0
                            num_rules = int(match.group(2))
                            task_data['pi_no_z3'] = test_acc
                            if 'num_rules' not in task_data or task_data['num_rules'] == 0:
                                task_data['num_rules'] = num_rules
                            break
                    elif 'No rules' in next_line or ('N/A' in next_line and 'Test:' not in next_line):
                        task_data['pi_no_z3'] = None
                        if 'num_rules' not in task_data:
                            task_data['num_rules'] = 0
                        break
            
            # Extract PI+Z3 test accuracy: "[3/3] WITH PI + Z3..."
            if '[3/3]' in line or ('WITH PI' in line and 'Z3' in line) or 'PI+Z3' in line.upper():
                for j in range(i+1, min(i+100, len(lines))):  # Increased search window
                    next_line = lines[j].strip()
                    # Skip progress bar lines
                    if '|' in next_line and '%' in next_line and 'Test:' not in next_line:
                        continue
                    if 'Test:' in next_line and 'rules' in next_line:
                        match = re.search(r'Test:\s*([\d.]+)%\s*\((\d+)\s*rules\)', next_line)
                        if match:
                            test_acc = float(match.group(1)) / 100.0
                            num_rules = int(match.group(2))
                            task_data['pi_z3'] = test_acc
                            task_data['num_rules'] = num_rules
                            break
                    elif 'No rules' in next_line or ('N/A' in next_line and 'Test:' not in next_line):
                        task_data['pi_z3'] = None
                        if 'num_rules' not in task_data:
                            task_data['num_rules'] = 0
                        break
    
    # Process all tasks found in SUMMARY
    for task_name, data in task_summaries.items():
        z3_delta = None
        if data.get('pi_z3') is not None and data.get('pi_no_z3') is not None:
            z3_delta = (data['pi_z3'] - data['pi_no_z3']) * 100
        
        pi_z3_val = data.get('pi_z3')
        results.append({
            'dataset': dataset,
            'problem': task_name,
            'num_rules': data.get('num_rules', 0),
            'train_accuracy': None,
            'test_accuracy': pi_z3_val,
            'test_f1': None,
            'no_pi': data.get('no_pi'),
            'pi_no_z3': data.get('pi_no_z3'),
            'pi_z3': pi_z3_val,
            'z3_delta': z3_delta,
            'time': data.get('time'),  # Use per-task time if available
            'status': 'PASS' if (pi_z3_val is not None and pi_z3_val > 0.5) else 'FAIL',
            'passed': pi_z3_val is not None and pi_z3_val > 0.5
        })
    
    # Handle last task if pattern matching was used
    if current_task and current_task not in task_summaries:
        z3_delta = None
        if task_data.get('pi_z3') is not None and task_data.get('pi_no_z3') is not None:
            z3_delta = (task_data['pi_z3'] - task_data['pi_no_z3']) * 100
        
        pi_z3_val = task_data.get('pi_z3')
        results.append({
            'dataset': dataset,
            'problem': current_task,
            'num_rules': task_data.get('num_rules', 0),
            'train_accuracy': None,
            'test_accuracy': pi_z3_val,
            'test_f1': None,
            'no_pi': task_data.get('no_pi'),
            'pi_no_z3': task_data.get('pi_no_z3'),
            'pi_z3': pi_z3_val,
            'z3_delta': z3_delta,
            'status': 'PASS' if (pi_z3_val is not None and pi_z3_val > 0.5) else 'FAIL',
            'passed': pi_z3_val is not None and pi_z3_val > 0.5
        })
    
    return results


def parse_test_all_output(output: str, dataset: str) -> List[Dict]:
 
    # Special handling for IP dataset
    if dataset == 'ip':
        return parse_ip_output(output, dataset)
    
    results = []
    lines = output.split('\n')
    
    # Find the table header
    header_line_idx = None
    for i, line in enumerate(lines):
        if 'Problem' in line and 'Rules' in line and 'Train Acc' in line:
            header_line_idx = i
            break
    
    if header_line_idx is None:
        return results
    
    # Parse table rows (skip header and separator)
    for i in range(header_line_idx + 2, len(lines)):
        line = lines[i].strip()
        if not line or line.startswith('-') or line.startswith('='):
            continue
        
        # Check if this is the summary line
        if 'Average' in line or 'Summary' in line:
            break
        
        # Parse row: Problem name, Rules, Train Acc, Test Acc, Test F1, Time (s), Status
        # Format: "problem_name              8        1.0000       0.8889       0.9000       2.14       PASS"
        # Or old format without time: "problem_name              8        1.0000       0.8889       0.9000       PASS"
        # Problem name can have spaces, so we need to find where numbers start
        parts = line.split()
        if len(parts) >= 6:
            try:
                # Find the first numeric part (Rules column)
                rules_idx = None
                for i, part in enumerate(parts):
                    try:
                        int(part)
                        rules_idx = i
                        break
                    except ValueError:
                        continue
                
                if rules_idx is None or rules_idx == 0:
                    continue
                
                # Problem name is everything before rules_idx
                problem_name = ' '.join(parts[:rules_idx])
                num_rules = int(parts[rules_idx])
                train_acc = float(parts[rules_idx + 1])
                test_acc = float(parts[rules_idx + 2])
                test_f1 = float(parts[rules_idx + 3])
                
                # Check if Time column exists (7 columns) or old format (6 columns)
                if len(parts) >= rules_idx + 6:
                    # New format with Time: Problem, Rules, Train Acc, Test Acc, Test F1, Time, Status
                    try:
                        time_val = float(parts[rules_idx + 4])
                        status = parts[rules_idx + 5] if len(parts) > rules_idx + 5 else 'UNKNOWN'
                    except (ValueError, IndexError):
                        # Old format without Time: Problem, Rules, Train Acc, Test Acc, Test F1, Status
                        time_val = None
                        status = parts[rules_idx + 4] if len(parts) > rules_idx + 4 else 'UNKNOWN'
                else:
                    # Old format
                    time_val = None
                    status = parts[rules_idx + 4] if len(parts) > rules_idx + 4 else 'UNKNOWN'
                
                results.append({
                    'dataset': dataset,
                    'problem': problem_name,
                    'num_rules': num_rules,
                    'train_accuracy': train_acc,
                    'test_accuracy': test_acc,
                    'test_f1': test_f1,
                    'time': time_val,  # Per-problem time if available
                    'status': status,
                    'passed': status in ['PASS', '[PASS]']
                })
            except (ValueError, IndexError) as e:
                if VERBOSE:
                    print(f"Warning: Could not parse line: {line}")
                continue
    
    # Extract average test accuracy and F1 if available
    avg_test_acc = None
    avg_test_f1 = None
    for line in lines:
        if 'Average Test Accuracy:' in line:
            match = re.search(r'Average Test Accuracy:\s*([\d.]+)', line)
            if match:
                avg_test_acc = float(match.group(1))
        if 'Average Test F1:' in line:
            match = re.search(r'Average Test F1:\s*([\d.]+)', line)
            if match:
                avg_test_f1 = float(match.group(1))
    
    # Add averages to each result for reference
    if avg_test_acc is not None:
        for r in results:
            r['dataset_avg_test_acc'] = avg_test_acc
            r['dataset_avg_test_f1'] = avg_test_f1
    
    return results


def run_test_script(dataset: str, script_path: str, random_seed: int = 42, timeout: int = 3600) -> tuple[str, int, float]:
    """
    Run a test_all.py script with a specific random seed.
    Returns (output, exit_code, elapsed_time).
    
    Note: Most test_all.py scripts have hardcoded random_state=42, so this
    parameter may not have an effect unless the scripts are modified to accept it.
    For now, we run multiple times and the seed variation will come from
    different execution contexts or we'll need to modify test scripts.
    """
    if not os.path.exists(script_path):
        return f"Error: Script not found: {script_path}", 1, 0.0
    
    # Change to the dataset directory
    dataset_dir = os.path.dirname(script_path)
    
    # Run the script with random seed as environment variable
    # (test scripts would need to check this, but for now we'll just run multiple times)
    env = os.environ.copy()
    env['PYGOL_RANDOM_SEED'] = str(random_seed)
    
    # Ensure we use the same Python interpreter (conda environment)
    # Try to find the pygol-z3-zendo conda environment Python
    python_exec = sys.executable
    # First try CONDA_PREFIX if we're in a conda environment
    if 'CONDA_PREFIX' in os.environ:
        conda_python = os.path.join(os.environ['CONDA_PREFIX'], 'bin', 'python')
        if os.path.exists(conda_python):
            python_exec = conda_python
    # Also try explicit path to pygol-z3-zendo environment
    pygol_z3_python = '/Users/godelmachine/miniforge3/envs/pygol-z3-zendo/bin/python'
    if os.path.exists(pygol_z3_python):
        python_exec = pygol_z3_python
    
    # Measure execution time
    start_time = time.time()
    
    # Use unbuffered Python (-u flag) to ensure output is captured immediately
    python_cmd = [python_exec, '-u', script_path]
    
    # Use Popen with process group to ensure we can kill all children on timeout
    # This is more reliable than subprocess.run() for killing child processes
    import signal
    
    # Create process with new process group (allows killing all children)
    # preexec_fn=os.setsid creates a new process group on Unix
    proc = subprocess.Popen(
        python_cmd,
        cwd=dataset_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
        bufsize=0,  # Unbuffered
        preexec_fn=os.setsid if hasattr(os, 'setsid') else None  # Create new process group (Unix only)
    )
    
    # Wait for process with timeout
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
        elapsed_time = time.time() - start_time
        output = stdout + stderr
        return output, proc.returncode, elapsed_time
    except subprocess.TimeoutExpired:
        # Timeout occurred - capture any output that was already printed
        elapsed_time = time.time() - start_time
        timeout_msg = f"[TIMEOUT] Script exceeded {timeout}s timeout after {elapsed_time:.1f}s"
        print(f"    {timeout_msg}", flush=True)
        
        # IMPORTANT: When communicate() raises TimeoutExpired, the pipes are still open
        # and we can read from them to get the output that was already printed!
        # This allows us to capture partial results from problems that completed before timeout.
        partial_output = ""
        try:
            # Read from stdout and stderr (they're still open after TimeoutExpired)
            if proc.stdout:
                try:
                    stdout_data = proc.stdout.read()
                    if stdout_data:
                        partial_output += stdout_data
                except:
                    pass
            if proc.stderr:
                try:
                    stderr_data = proc.stderr.read()
                    if stderr_data:
                        partial_output += stderr_data
                except:
                    pass
        except Exception as read_error:
            # If reading fails, that's okay - we'll work with what we have
            if VERBOSE:
                print(f"    Warning: Could not read partial output: {read_error}", flush=True)
        
        # Kill the entire process group (kills process and all children)
        try:
            if hasattr(os, 'killpg') and hasattr(os, 'getpgid'):
                # Unix: kill entire process group
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            else:
                # Windows or fallback: just kill the process
                proc.kill()
            proc.wait(timeout=5)  # Wait for it to die
        except (OSError, ProcessLookupError, subprocess.TimeoutExpired):
            # Process already dead or couldn't kill it
            pass
        except Exception as kill_error:
            if VERBOSE:
                print(f"    Warning: Error killing timed-out process: {kill_error}", flush=True)
        
        # Return partial output with timeout indicator
        # The parser can extract results from completed problems (those that printed
        # their results before the timeout). Problems that didn't complete won't have
        # results, which is expected.
        if partial_output:
            output_with_timeout = partial_output + f"\n[TIMEOUT] Trial exceeded {timeout}s limit - partial results only\n"
            print(f"    [INFO] Captured partial output ({len(partial_output)} chars) - will parse completed problems", flush=True)
        else:
            # No output captured - this shouldn't happen, but handle it gracefully
            output_with_timeout = f"[TIMEOUT] Trial exceeded {timeout}s limit - no output captured\n"
            print(f"    [WARNING] No partial output captured", flush=True)
        
        return output_with_timeout, -1, elapsed_time


def format_dataset_markdown(dataset: str, results: List[Dict]) -> str:
    """Format results for a single dataset as Markdown."""
    dataset_name = dataset.upper()
    md = f"# PyGol+Z3 Experimental Results - {dataset_name}\n\n"
    md += f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    num_trials = results[0].get('num_trials', NUM_TRIALS) if results else NUM_TRIALS
    md += f"Results from {num_trials} trial(s) per problem.\n\n"
    md += "## Main Results\n\n"
    md += f"## {dataset_name}\n\n"
    
    # Special formatting for IP dataset (shows No PI, PI only, PI+Z3, Z3 Δ, Time, Rules)
    if dataset == 'ip':
        md += "| Problem | No PI (%) | PI Only (%) | PI+Z3 (%) | Z3 Δ (%) | Time (s) | Rules |\n"
        md += "|---------|-----------|-------------|-----------|----------|----------|-------|\n"
        for r in results:
            no_pi = r.get('no_pi')
            pi_no_z3 = r.get('pi_no_z3')
            pi_z3 = r.get('pi_z3')
            z3_delta = r.get('z3_delta')
            time_mean = r.get('time_mean', None)
            time_se = r.get('time_se', None)
            num_rules = r.get('num_rules', 0)
            num_rules_se = r.get('num_rules_se', 0.0)
            num_trials = r.get('num_trials', 1)
            
            # Get standard errors for IP-specific fields
            no_pi_se = r.get('no_pi_se', None)
            pi_no_z3_se = r.get('pi_no_z3_se', None)
            pi_z3_se = r.get('pi_z3_se', None)
            z3_delta_se = r.get('z3_delta_se', None)
            
            # Format No PI with SE
            if no_pi is not None:
                if no_pi_se is not None and num_trials > 1:
                    no_pi_str = f"{no_pi*100:.2f} ± {no_pi_se*100:.2f}"
                else:
                    no_pi_str = f"{no_pi*100:.2f}"
            else:
                no_pi_str = "N/A"
            
            # Format PI Only with SE
            if pi_no_z3 is not None:
                if pi_no_z3_se is not None and num_trials > 1:
                    pi_no_z3_str = f"{pi_no_z3*100:.2f} ± {pi_no_z3_se*100:.2f}"
                else:
                    pi_no_z3_str = f"{pi_no_z3*100:.2f}"
            else:
                pi_no_z3_str = "N/A"
            
            # Format PI+Z3 with SE
            if pi_z3 is not None:
                if pi_z3_se is not None and num_trials > 1:
                    pi_z3_str = f"{pi_z3*100:.2f} ± {pi_z3_se*100:.2f}"
                else:
                    pi_z3_str = f"{pi_z3*100:.2f}"
            else:
                pi_z3_str = "N/A"
            
            # Format Z3 Δ with SE
            if z3_delta is not None:
                if z3_delta_se is not None and num_trials > 1:
                    z3_delta_str = f"{z3_delta:+.2f} ± {z3_delta_se:.2f}"
                else:
                    z3_delta_str = f"{z3_delta:+.2f}"
            else:
                z3_delta_str = "N/A"
            
            # Format time with SE
            if time_mean is not None:
                if time_se is not None and time_se > 0 and num_trials > 1:
                    time_str = f"{time_mean:.2f} ± {time_se:.2f}"
                else:
                    time_str = f"{time_mean:.2f}"
            else:
                time_str = "N/A"
            
            # Format rules with SE
            if num_rules_se > 0 and num_trials > 1:
                rules_str = f"{num_rules:.1f} ± {num_rules_se:.1f}"
            else:
                rules_str = f"{num_rules}"
            
            md += f"| {r['problem']} | {no_pi_str} | {pi_no_z3_str} | {pi_z3_str} | {z3_delta_str} | {time_str} | {rules_str} |\n"
        
        # Calculate average (only for PI+Z3)
        if results:
            pi_z3_values = [r.get('pi_z3') for r in results if r.get('pi_z3') is not None]
            if pi_z3_values:
                avg_pi_z3 = sum(pi_z3_values) / len(pi_z3_values)
                md += f"\n**Average PI+Z3 Accuracy:** {avg_pi_z3*100:.2f}%\n\n"
        return md
    
    # Check if we have standard errors (multiple trials)
    has_se = any('test_accuracy_se' in r for r in results)
    
    if has_se:
        # Match previous format: Problem | Test Accuracy (%) | Train Accuracy (%) | Time (s) | Rules | Config
        md += "| Problem | Test Accuracy (%) | Train Accuracy (%) | Time (s) | Rules | Config |\n"
        md += "|---------|---------------|----------------|----------|-------|--------|\n"
        for r in results:
            test_acc_se = r.get('test_accuracy_se', 0.0)
            train_acc_se = r.get('train_accuracy_se', 0.0)
            time_mean = r.get('time_mean', None)
            time_se = r.get('time_se', None)
            num_rules_se = r.get('num_rules_se', 0.0)
            
            # Format test accuracy with SE (always show ± for consistency, even if SE = 0.00)
            # Values are in decimal (0.0-1.0), multiply by 100 for percentage display
            # Check if we have SE data (from multiple trials)
            if 'test_accuracy_se' in r and r.get('num_trials', 1) > 1:
                test_acc_se_val = r.get('test_accuracy_se', 0.0) * 100
                test_acc_val = r['test_accuracy'] * 100
                test_acc_str = f"{test_acc_val:.2f} ± {test_acc_se_val:.2f}"
            else:
                test_acc_val = r['test_accuracy'] * 100
                test_acc_str = f"{test_acc_val:.2f}"
            
            # Format train accuracy with SE (always show ± for consistency, even if SE = 0.00)
            # For IP, train accuracy is None (not reported), so show N/A
            if dataset == 'ip' and (r.get('train_accuracy') is None or r.get('train_accuracy', 0.0) == 0.0):
                train_acc_str = "N/A"
            elif 'train_accuracy_se' in r and r.get('num_trials', 1) > 1:
                train_acc_se_val = r.get('train_accuracy_se', 0.0) * 100
                train_acc_val = r['train_accuracy'] * 100
                train_acc_str = f"{train_acc_val:.2f} ± {train_acc_se_val:.2f}"
            else:
                train_acc_val = r['train_accuracy'] * 100
                train_acc_str = f"{train_acc_val:.2f}"
            
            # Format time with SE
            if time_mean is not None:
                if time_se is not None and time_se > 0:
                    time_str = f"{time_mean:.2f} ± {time_se:.2f}"
                else:
                    time_str = f"{time_mean:.2f}"
            else:
                time_str = "N/A"
            
            # Format rules with SE
            if num_rules_se > 0:
                rules_str = f"{r['num_rules']:.1f} ± {num_rules_se:.1f}"
            else:
                rules_str = f"{r['num_rules']}"
            
            # Config: determine from dataset config or use default
            config = r.get('config', 'Z3')
            if isinstance(config, dict):
                if config.get('pi_enabled') and config.get('z3_enabled'):
                    config = 'PI+Z3'
                elif config.get('z3_enabled'):
                    config = 'Z3'
                elif config.get('pi_enabled'):
                    config = 'PI'
                else:
                    config = 'Base'
            
            md += f"| {r['problem']} | {test_acc_str} | {train_acc_str} | {time_str} | {rules_str} | {config} |\n"
    else:
        # Match previous format for single trial
        md += "| Problem | Test Accuracy (%) | Train Accuracy (%) | Time (s) | Rules | Config |\n"
        md += "|---------|---------------|----------------|----------|-------|--------|\n"
        for r in results:
            time_mean = r.get('time_mean', None)
            time_str = f"{time_mean:.2f}" if time_mean is not None else "N/A"
            config = r.get('config', 'Z3')
            if isinstance(config, dict):
                if config.get('pi_enabled') and config.get('z3_enabled'):
                    config = 'PI+Z3'
                elif config.get('z3_enabled'):
                    config = 'Z3'
                elif config.get('pi_enabled'):
                    config = 'PI'
                else:
                    config = 'Base'
            md += f"| {r['problem']} | {r['test_accuracy']*100:.2f} | {r['train_accuracy']*100:.2f} | {time_str} | {r['num_rules']} | {config} |\n"
    
    # Add average if available
    if results and 'dataset_avg_test_acc' in results[0]:
        avg_acc = results[0]['dataset_avg_test_acc']
        avg_f1 = results[0]['dataset_avg_test_f1']
        md += f"\n**Average Test Accuracy:** {avg_acc*100:.2f}%  \n"
        md += f"**Average Test F1:** {avg_f1*100:.2f}%\n\n"
    else:
        # Calculate from results
        if results:
            avg_acc = sum(r['test_accuracy'] for r in results) / len(results)
            avg_f1 = sum(r['test_f1'] for r in results) / len(results)
            md += f"\n**Average Test Accuracy:** {avg_acc*100:.2f}%  \n"
            md += f"**Average Test F1:** {avg_f1*100:.2f}%\n\n"
    
    return md


def format_results_markdown(all_results: Dict[str, List[Dict]]) -> str:
    """Format all results as Markdown tables."""
    md = "# PyGol+Z3 Experimental Results\n\n"
    md += f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    md += "Results from individual test_all.py scripts for each dataset.\n\n"
    
    # Dataset order
    dataset_order = ['geometry0', 'geometry1', 'ip', 'geometry3', 'geometry2']
    
    for dataset in dataset_order:
        if dataset not in all_results or not all_results[dataset]:
            continue
        
        results = all_results[dataset]
        dataset_name = dataset.upper()
        
        md += f"## {dataset_name}\n\n"
        md += "| Problem | Rules | Train Acc (%) | Test Acc (%) | Test F1 (%) | Status |\n"
        md += "|---------|-------|-----------|----------|---------|--------|\n"
        
        for r in results:
            status = r.get('status', 'UNKNOWN')
            md += f"| {r['problem']} | {r['num_rules']} | {r['train_accuracy']*100:.2f} | {r['test_accuracy']*100:.2f} | {r['test_f1']*100:.2f} | {status} |\n"
        
        # Add average if available
        if results and 'dataset_avg_test_acc' in results[0]:
            avg_acc = results[0]['dataset_avg_test_acc']
            avg_f1 = results[0]['dataset_avg_test_f1']
            md += f"\n**Average Test Accuracy:** {avg_acc*100:.2f}%  \n"
            md += f"**Average Test F1:** {avg_f1*100:.2f}%\n\n"
        else:
            # Calculate from results
            if results:
                avg_acc = sum(r['test_accuracy'] for r in results) / len(results)
                avg_f1 = sum(r['test_f1'] for r in results) / len(results)
                md += f"\n**Average Test Accuracy:** {avg_acc:.4f}  \n"
                md += f"**Average Test F1:** {avg_f1:.4f}\n\n"
    
    return md


def format_results_text(all_results: Dict[str, List[Dict]]) -> str:
    """Format all results as plain text tables."""
    text = "PyGol+Z3 Experimental Results\n"
    text += "=" * 80 + "\n"
    text += f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    
    dataset_order = ['geometry0', 'geometry1', 'ip', 'geometry3', 'geometry2']
    
    for dataset in dataset_order:
        if dataset not in all_results or not all_results[dataset]:
            continue
        
        results = all_results[dataset]
        dataset_name = dataset.upper()
        
        text += f"\n{dataset_name}\n"
        text += "-" * 80 + "\n"
        text += f"{'Problem':<25} {'Rules':<8} {'Train Acc':<12} {'Test Acc':<12} {'Test F1':<12} {'Status':<10}\n"
        text += "-" * 80 + "\n"
        
        for r in results:
            status = r.get('status', 'UNKNOWN')
            text += f"{r['problem']:<25} {r['num_rules']:<8} {r['train_accuracy']*100:<15.2f} {r['test_accuracy']*100:<15.2f} {r['test_f1']*100:<15.2f} {status:<10}\n"
        
        # Add average
        if results:
            avg_acc = sum(r['test_accuracy'] for r in results) / len(results)
            avg_f1 = sum(r['test_f1'] for r in results) / len(results)
            text += f"\nAverage Test Accuracy: {avg_acc*100:.2f}%\n"
            text += f"Average Test F1: {avg_f1*100:.2f}%\n\n"
    
    return text


def check_dataset_complete(dataset: str, expected_problems: List[str] = None) -> bool:
    """
    Check if a dataset's output file exists and contains results for all expected problems.
    
    Args:
        dataset: Dataset name (e.g., 'geometry1')
        expected_problems: List of expected problem names. If None, will try to infer from file.
    
    Returns:
        True if dataset appears complete, False otherwise
    """
    output_file = os.path.join(_experiments_dir, f'pygol_z3_results_{dataset}.md')
    
    if not os.path.exists(output_file):
        return False
    
    # Read the file and check for results
    try:
        with open(output_file, 'r') as f:
            content = f.read()
        
        # Check if file has a results table (could be "Test Acc" or "Test Accuracy")
        if '| Problem |' not in content:
            return False
        
        # Check for test accuracy column (could be "Test Acc" or "Test Accuracy")
        # For IP, check for "No PI" or "PI+Z3" columns instead
        if dataset == 'ip':
            if '| No PI' not in content and '| PI+Z3' not in content and '| PI Only' not in content:
                return False
        else:
            if '| Test Acc' not in content and '| Test Accuracy' not in content:
                return False
        
        # Count number of result rows (non-header, non-separator rows)
        lines = content.split('\n')
        result_count = 0
        in_table = False
        found_problems = set()
        
        for line in lines:
            if '| Problem |' in line:
                in_table = True
                continue
            if in_table and line.strip().startswith('|') and not line.strip().startswith('|---'):
                # Check if it's a data row (has problem name and numbers)
                parts = [p.strip() for p in line.split('|') if p.strip()]
                if len(parts) >= 4:
                    try:
                        # First part should be problem name, 4th part should be test accuracy
                        problem_name = parts[0]
                        test_acc_str = parts[3].replace('%', '').replace('±', '').strip()
                        # Try to parse test accuracy (could be "0.8889" or "88.89% ± 1.22%")
                        # Extract first number
                        import re
                        match = re.search(r'[\d.]+', test_acc_str)
                        if match:
                            float(match.group())
                            result_count += 1
                            found_problems.add(problem_name)
                    except (ValueError, IndexError, AttributeError):
                        pass
        
        # If we have expected problems, check if we have results for all
        if expected_problems:
            # Normalize problem names (remove spaces, case-insensitive, handle variations)
            def normalize_name(name):
                name = name.lower().replace(' ', '').replace('_', '').replace('-', '')
                # Handle common variations
                if '3d' in name or '3d' in name:
                    name = name.replace('3d', '')
                return name
            
            expected_normalized = {normalize_name(p) for p in expected_problems}
            found_normalized = {normalize_name(p) for p in found_problems}
            # Check if all expected problems are found (allow some flexibility)
            matches = expected_normalized.intersection(found_normalized)
            # Consider complete if we have at least 80% of expected problems
            return len(matches) >= max(1, int(len(expected_problems) * 0.8))
        
        # Otherwise, consider it complete if we have at least one result
        return result_count > 0
        
    except Exception as e:
        if VERBOSE:
            print(f"Warning: Error checking {dataset} completeness: {e}")
        return False


def test_ip_parser():
    """Test function to verify IP parser works correctly."""
    sample_output = """InfluencePropagation (IP): PI + Z3 Requirement Benchmark

Testing 5 tasks with progressive difficulty:
  1. ip1_active      - Simple (2 literals) - Baseline
  2. ip2_active      - Multi-hop (3 literals) - Requires PI
  3. ip3_active      - Multi-hop (3 literals) - Requires PI
  4. ip3_threshold   - Multi-hop + numerical - Requires PI + Z3
  5. ip4_high_score  - Multi-hop + aggregate - Requires PI + Z3

Testing: ip1_active
Data: 51 train (35 pos), 22 test (15 pos)

[1/3] WITHOUT PI (max_literals=2)...
  Test: 100.00% (1 rules)

[2/3] WITH PI, NO Z3 (max_literals=4, pi=True)...
  Test: 100.00% (1 rules)

[3/3] WITH PI + Z3 (full integration)...
  Test: 100.00% (2 rules)

SUMMARY:
  NO PI:      100.00%
  PI, NO Z3:  100.00%
  PI + Z3:    100.00% (Z3 benefit: +0.0%)

Testing: ip2_active
Data: 51 train (35 pos), 22 test (15 pos)

[1/3] WITHOUT PI (max_literals=2)...
  No rules -> N/A

[2/3] WITH PI, NO Z3 (max_literals=4, pi=True)...
  Test: 89.70% (2 rules)

[3/3] WITH PI + Z3 (full integration)...
  Test: 89.70% (2 rules)

SUMMARY:
  NO PI:      N/A
  PI, NO Z3:  89.70%
  PI + Z3:    89.70% (Z3 benefit: +0.0%)

Testing: ip3_active
Data: 51 train (35 pos), 22 test (15 pos)

[1/3] WITHOUT PI (max_literals=2)...
  No rules -> N/A

[2/3] WITH PI, NO Z3 (max_literals=4, pi=True)...
  Test: 60.90% (2 rules)

[3/3] WITH PI + Z3 (full integration)...
  Test: 82.60% (3 rules)

SUMMARY:
  NO PI:      N/A
  PI, NO Z3:  60.90%
  PI + Z3:    82.60% (Z3 benefit: +21.7%)
"""

    print("Testing IP parser with sample output...")

    results = parse_ip_output(sample_output, 'ip')

    print(f"\n[OK] Parsed {len(results)} tasks\n")

    for r in results:
        print(f"Task: {r['problem']}")
        no_pi = r.get('no_pi')
        pi_no_z3 = r.get('pi_no_z3')
        pi_z3 = r.get('pi_z3')
        z3_delta = r.get('z3_delta')
        
        print(f"  No PI:      {no_pi*100:.2f}%" if no_pi is not None else "  No PI:      N/A")
        print(f"  PI+NoZ3:    {pi_no_z3*100:.2f}%" if pi_no_z3 is not None else "  PI+NoZ3:    N/A")
        print(f"  PI+Z3:      {pi_z3*100:.2f}%" if pi_z3 is not None else "  PI+Z3:      N/A")
        print(f"  Z3 Δ:       {z3_delta:+.2f}%" if z3_delta is not None else "  Z3 Δ:       N/A")
        print(f"  Rules:      {r.get('num_rules', 0)}")
        print(f"  Status:     {r.get('status', 'UNKNOWN')}")
        print()

    # Verify expected values
    expected = {
        'ip1_active': {'no_pi': 1.0, 'pi_no_z3': 1.0, 'pi_z3': 1.0, 'z3_delta': 0.0},
        'ip2_active': {'no_pi': None, 'pi_no_z3': 0.897, 'pi_z3': 0.897, 'z3_delta': 0.0},
        'ip3_active': {'no_pi': None, 'pi_no_z3': 0.609, 'pi_z3': 0.826, 'z3_delta': 21.7},
    }

    print("Verification:")
    all_passed = True
    for task_name, expected_vals in expected.items():
        result = next((r for r in results if r['problem'] == task_name), None)
        if result:
            task_passed = True
            for key, expected_val in expected_vals.items():
                actual_val = result.get(key)
                if expected_val is None:
                    if actual_val is not None:
                        print(f"[ERROR] {task_name}.{key}: Expected None, got {actual_val}")
                        all_passed = False
                        task_passed = False
                else:
                    if actual_val is None:
                        print(f"[ERROR] {task_name}.{key}: Expected {expected_val}, got None")
                        all_passed = False
                        task_passed = False
                    elif abs(actual_val - expected_val) > 0.01:
                        print(f"[ERROR] {task_name}.{key}: Expected {expected_val}, got {actual_val}")
                        all_passed = False
                        task_passed = False
            if task_passed:
                print(f"[OK] {task_name}: All values match")
        else:
            print(f"[ERROR] {task_name}: Not found in results")
            all_passed = False

    print()
    if all_passed:
        print("[OK] ALL TESTS PASSED - Parser is working correctly!")
    else:
        print("[ERROR] SOME TESTS FAILED - Parser needs fixing")
    
    return all_passed


def main():
    """Main function to run all test scripts and aggregate results."""
    
    # Define test scripts for each dataset
    # Order: Fast datasets first, slow ones last (geometry2 is slowest)
    test_scripts = {
        'geometry0': os.path.join(_experiments_dir, 'geometry0', 'test_geometry0_all.py'),
        'geometry1': os.path.join(_experiments_dir, 'geometry1', 'test_geometry1_all.py'),
        'ip': os.path.join(_experiments_dir, 'ip', 'test_ip_all.py'),
        'geometry3': os.path.join(_experiments_dir, 'geometry3', 'test_geometry3_all.py'),
        'geometry2': os.path.join(_experiments_dir, 'geometry2', 'test_geometry2_all.py'),
    }
    
    # Expected number of problems per dataset (for completeness check)
    expected_problems = {
        'geometry0': ['interval', 'halfplane'],
        'geometry1': ['3D Halfplane', 'Conjunction', 'Multiple Halfplanes', '3D Interval'],
        'geometry2': ['left_of', 'closer_than', 'touching', 'inside', 'overlapping', 
                      'between', 'adjacent', 'aligned', 'surrounds', 'near_corner'],
        'geometry3': ['in_circle', 'in_ellipse', 'hyperbola_side', 'xy_less_than', 'quad_strip',
                      'union_halfplanes', 'circle_or_box', 'piecewise', 'fallback_region',
                      'donut', 'lshape', 'above_parabola', 'sinusoidal', 'crescent'],
        'ip': ['ip1_active', 'ip2_active', 'ip3_active', 'ip3_threshold', 'ip4_high_score'],
    }
    
    all_results = {}
    
    print("Running PyGol+Z3 experiments on all datasets...", flush=True)
    print(f"Running {NUM_TRIALS} trial(s) per problem with different random seeds", flush=True)
    
    # Run each test script
    for dataset, script_path in test_scripts.items():
        if not os.path.exists(script_path):
            print(f"\n[SKIP] Skipping {dataset}: Script not found: {script_path}")
            continue
        
        # Check if dataset is already complete
        expected = expected_problems.get(dataset, None)
        if check_dataset_complete(dataset, expected):
            print(f"\n[{dataset.upper()}] SKIPPING - Results already exist and appear complete", flush=True)
            print(f"  Output file: pygol_z3_results_{dataset}.md", flush=True)
            # Try to load existing results from the file
            output_file = os.path.join(_experiments_dir, f'pygol_z3_results_{dataset}.md')
            try:
                with open(output_file, 'r') as f:
                    content = f.read()
                # Parse existing results
                if dataset == 'ip':
                    existing_results = parse_ip_output(content, dataset)
                else:
                    existing_results = parse_test_all_output(content, dataset)
                if existing_results:
                    all_results[dataset] = existing_results
                    print(f"  [OK] Loaded {len(existing_results)} problems from existing file")
                else:
                    print(f"  Warning: Could not parse results, but file exists - skipping anyway")
                continue  # Always skip if file exists and appears complete
            except Exception as e:
                if VERBOSE:
                    print(f"  Warning: Could not read existing file: {e}")
                # Still skip if file exists and appears complete (parsing failure doesn't mean incomplete)
                continue
        
        # Adjust trials per dataset:
        # - geometry2 and geometry3: 3 trials (they take too long)
        # - ip: 5 trials (faster, can do more)
        # - geometry0 and geometry1: 5 trials (default)
        if dataset in ['geometry2', 'geometry3']:
            num_trials_for_dataset = 3
        elif dataset == 'ip':
            num_trials_for_dataset = 5
        else:
            num_trials_for_dataset = NUM_TRIALS
        
        print(f"\n[{dataset.upper()}] Running {os.path.basename(script_path)} ({num_trials_for_dataset} trials)...", flush=True)
        
        # Run multiple trials with different random seeds
        all_trial_results = {}  # problem_name -> list of results across trials
        
        for trial in range(num_trials_for_dataset):
            random_seed = 42 + trial  # Use seeds 42-44 for geometry2/3 (3 trials), 42-46 for others (5 trials)
            if num_trials_for_dataset > 1:
                print(f"  Trial {trial + 1}/{num_trials_for_dataset} (seed={random_seed})...", flush=True)
            
            # Use shorter timeout for IP (it has 5 tasks × 3 configs = 15 runs per trial)
            # Geometry2 has 10 problems, expect ~1.5 hours per trial, but can take longer
            # Use 4 hours to allow completion of all problems (was 3 hours, but that was timing out)
            if dataset == 'ip':
                timeout_seconds = 1800  # 30 min for IP
            elif dataset == 'geometry2':
                timeout_seconds = 14400  # 4 hours for geometry2 (10 problems × ~12-15 min each = 2-2.5 hours, +1.5 hour safety)
            else:
                timeout_seconds = 3600  # 1 hour for others
            output, exit_code, elapsed_time = run_test_script(dataset, script_path, random_seed=random_seed, timeout=timeout_seconds)
            
            if exit_code != 0:
                print(f"    [WARNING] Script exited with code {exit_code}", flush=True)
                if VERBOSE or exit_code != 0:  # Always show errors, even if not verbose
                    print(f"    Output preview (first 500 chars):", flush=True)
                    print(f"    {output[:500]}", flush=True)
            
            # Parse output for this trial
            if dataset == 'ip':
                trial_results = parse_ip_output(output, dataset)
            else:
                trial_results = parse_test_all_output(output, dataset)
            
            # Add time to each result
            # For IP: use per-task time if parsed from output, otherwise divide total time
            # For other datasets: divide total time by number of problems
            num_problems = len(trial_results) if trial_results else 1
            time_per_problem = elapsed_time / num_problems if num_problems > 0 else elapsed_time
            
            # Aggregate results by problem name
            for result in trial_results:
                problem_name = result['problem']
                # Use per-task time if available (from SUMMARY), otherwise use divided time
                if result.get('time') is None:
                    result['time'] = time_per_problem
                if problem_name not in all_trial_results:
                    all_trial_results[problem_name] = []
                all_trial_results[problem_name].append(result)
        
        # Aggregate across trials (calculate mean ± SE)
        aggregated_results = []
        for problem_name, trial_results_list in all_trial_results.items():
            if not trial_results_list:
                continue
            
            # Calculate statistics across trials
            test_accs = [r['test_accuracy'] for r in trial_results_list if r.get('test_accuracy') is not None]
            train_accs = [r['train_accuracy'] for r in trial_results_list if r.get('train_accuracy') is not None]
            test_f1s = [r.get('test_f1', 0.0) for r in trial_results_list if r.get('test_f1') is not None]
            num_rules_list = [r['num_rules'] for r in trial_results_list]
            times = [r.get('time', None) for r in trial_results_list if r.get('time') is not None]
            
            # Use first trial's result as base, update with aggregated stats
            base_result = trial_results_list[0].copy()
            
            # For IP dataset, aggregate special fields
            if dataset == 'ip':
                no_pi_list = [r.get('no_pi') for r in trial_results_list if r.get('no_pi') is not None]
                pi_no_z3_list = [r.get('pi_no_z3') for r in trial_results_list if r.get('pi_no_z3') is not None]
                pi_z3_list = [r.get('pi_z3') for r in trial_results_list if r.get('pi_z3') is not None]
                
                if no_pi_list:
                    base_result['no_pi'] = np.mean(no_pi_list)
                    base_result['no_pi_se'] = np.std(no_pi_list) / np.sqrt(len(no_pi_list)) if len(no_pi_list) > 1 else 0.0
                if pi_no_z3_list:
                    base_result['pi_no_z3'] = np.mean(pi_no_z3_list)
                    base_result['pi_no_z3_se'] = np.std(pi_no_z3_list) / np.sqrt(len(pi_no_z3_list)) if len(pi_no_z3_list) > 1 else 0.0
                if pi_z3_list:
                    base_result['pi_z3'] = np.mean(pi_z3_list)
                    base_result['pi_z3_se'] = np.std(pi_z3_list) / np.sqrt(len(pi_z3_list)) if len(pi_z3_list) > 1 else 0.0
                    base_result['test_accuracy'] = np.mean(pi_z3_list)  # Use PI+Z3 as main test accuracy
                
                # Recalculate Z3 Δ from aggregated means (always calculate if both values exist, even if result is 0.0)
                if base_result.get('pi_z3') is not None and base_result.get('pi_no_z3') is not None:
                    base_result['z3_delta'] = (base_result['pi_z3'] - base_result['pi_no_z3']) * 100
                    # Calculate Z3 Δ SE using error propagation: SE(z3_delta) = sqrt(SE(pi_z3)^2 + SE(pi_no_z3)^2)
                    pi_z3_se_val = base_result.get('pi_z3_se', 0.0)
                    pi_no_z3_se_val = base_result.get('pi_no_z3_se', 0.0)
                    base_result['z3_delta_se'] = np.sqrt(pi_z3_se_val**2 + pi_no_z3_se_val**2) * 100
                else:
                    base_result['z3_delta'] = None
                    base_result['z3_delta_se'] = None
            else:
                base_result['test_accuracy_mean'] = np.mean(test_accs) if test_accs else 0.0
                base_result['test_accuracy_se'] = np.std(test_accs) / np.sqrt(len(test_accs)) if len(test_accs) > 1 else 0.0
                base_result['train_accuracy_mean'] = np.mean(train_accs) if train_accs else 0.0
                base_result['train_accuracy_se'] = np.std(train_accs) / np.sqrt(len(train_accs)) if len(train_accs) > 1 else 0.0
                base_result['test_f1_mean'] = np.mean(test_f1s) if test_f1s else 0.0
                base_result['test_f1_se'] = np.std(test_f1s) / np.sqrt(len(test_f1s)) if len(test_f1s) > 1 else 0.0
                base_result['test_accuracy'] = base_result['test_accuracy_mean']
                base_result['train_accuracy'] = base_result['train_accuracy_mean']
                base_result['test_f1'] = base_result['test_f1_mean']
            
            base_result['num_rules_mean'] = np.mean(num_rules_list) if num_rules_list else 0.0
            base_result['num_rules_se'] = np.std(num_rules_list) / np.sqrt(len(num_rules_list)) if len(num_rules_list) > 1 else 0.0
            base_result['num_trials'] = len(trial_results_list)
            
            # Time statistics
            if times:
                base_result['time_mean'] = np.mean(times)
                base_result['time_se'] = np.std(times) / np.sqrt(len(times)) if len(times) > 1 else 0.0
            else:
                base_result['time_mean'] = None
                base_result['time_se'] = None
            
            # Update main fields to use means (already done for IP above)
            if dataset != 'ip':
                base_result['test_accuracy'] = base_result.get('test_accuracy_mean', base_result.get('test_accuracy', 0.0))
                base_result['train_accuracy'] = base_result.get('train_accuracy_mean', base_result.get('train_accuracy', 0.0))
                base_result['test_f1'] = base_result.get('test_f1_mean', base_result.get('test_f1', 0.0))
            base_result['num_rules'] = int(round(base_result['num_rules_mean']))
            
            # Preserve config if available, otherwise default to Z3
            if 'config' not in base_result:
                base_result['config'] = 'Z3'
            
            aggregated_results.append(base_result)
        
        if aggregated_results:
            all_results[dataset] = aggregated_results
            print(f"  [OK] Aggregated {len(aggregated_results)} problems across {NUM_TRIALS} trial(s)")
            if VERBOSE:
                for r in aggregated_results:
                    if NUM_TRIALS > 1:
                        print(f"    - {r['problem']}: {r['test_accuracy']:.2%} ± {r['test_accuracy_se']:.2%} test acc")
                    else:
                        print(f"    - {r['problem']}: {r['test_accuracy']:.2%} test acc")
            
            # Write per-dataset file immediately after completion
            dataset_md = format_dataset_markdown(dataset, aggregated_results)
            dataset_md_file = os.path.join(_experiments_dir, f'pygol_z3_results_{dataset}.md')
            with open(dataset_md_file, 'w') as f:
                f.write(dataset_md)
            print(f"  [OK] Wrote results to {os.path.basename(dataset_md_file)}")
        else:
            print(f"  [ERROR] No results parsed (check output format)", flush=True)
            # Always show output preview if parsing failed
            print(f"  Output preview (first 1000 chars):", flush=True)
            print(f"  {output[:1000]}", flush=True)
    
    # Generate output files
    print("\nGenerating output files...")
    
    # Per-dataset markdown files (for easy access)
    for dataset, results in all_results.items():
        if results:
            dataset_md = format_dataset_markdown(dataset, results)
            dataset_md_file = os.path.join(_experiments_dir, f'pygol_z3_results_{dataset}.md')
            with open(dataset_md_file, 'w') as f:
                f.write(dataset_md)
            print(f"  [OK] {dataset.upper()} Markdown: {dataset_md_file}")
    
    # Combined markdown format
    md_content = format_results_markdown(all_results)
    md_file = os.path.join(_experiments_dir, 'pygol_z3_results.md')
    with open(md_file, 'w') as f:
        f.write(md_content)
    print(f"  [OK] Combined Markdown: {md_file}")
    
    # Plain text format
    text_content = format_results_text(all_results)
    text_file = os.path.join(_experiments_dir, 'pygol_z3_results.txt')
    with open(text_file, 'w') as f:
        f.write(text_content)
    print(f"  [OK] Plain text: {text_file}")
    
    # JSON format
    json_file = os.path.join(_experiments_dir, 'pygol_z3_results.json')
    with open(json_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"  [OK] JSON: {json_file}")
    
    # Summary
    print("\nSummary:")
    total_problems = sum(len(results) for results in all_results.values())
    total_passed = sum(sum(1 for r in results if r.get('passed', False)) for results in all_results.values())
    print(f"  Total problems: {total_problems}")
    print(f"  Passed: {total_passed}")
    print(f"  Failed: {total_problems - total_passed}")
    
    print("\nDone!")


if __name__ == '__main__':
    import sys
    # Allow running parser test: python run_all_pygol_z3.py --test-parser
    if len(sys.argv) > 1 and sys.argv[1] == '--test-parser':
        test_ip_parser()
    else:
        main()

