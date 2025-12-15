#!/usr/bin/env python3
"""
Convert all existing datasets to NumSynth format for fair comparison.

This script:
1. Reads existing example files from geometry0, geometry1, geometry2, geometry3, ip
2. Converts them to NumSynth format (pos/neg wrapper)
3. Creates BK files compatible with NumSynth
4. Creates bias files for NumSynth
5. Organizes everything in numsynth_datasets/ folder
"""

import os
import sys
import re
import shutil
from pathlib import Path

# Base paths
_current_file = os.path.abspath(__file__)
_numsynth_comparison_dir = os.path.dirname(_current_file)  # numsynth_comparison folder
_experiments_dir = os.path.dirname(_numsynth_comparison_dir)  # Parent experiments folder
_output_dir = os.path.join(_numsynth_comparison_dir, 'numsynth_datasets')  # In numsynth_comparison folder
# numsynth-main is in numsynth_comparison folder, not experiments folder
_numsynth_main = os.path.join(_numsynth_comparison_dir, 'numsynth-main', 'ilp-experiments', 'ilpexp', 'problem', 'geometry')


def ensure_dir(path):
    """Create directory if it doesn't exist"""
    os.makedirs(path, exist_ok=True)


def convert_examples_file(input_file, output_file):
    """
    Convert example file to NumSynth format.
    Handles both formats:
    - pos(pred(...)). / neg(pred(...)). (already correct)
    - pred(...). / neg(pred(...)). (needs conversion)
    """
    if not os.path.exists(input_file):
        return False
    
    with open(input_file, 'r') as f:
        lines = f.readlines()
    
    with open(output_file, 'w') as f:
        for line in lines:
            line = line.strip()
            if not line or line.startswith('%'):
                continue
            
            # Already in pos/neg format
            if line.startswith('pos(') or line.startswith('neg('):
                f.write(line + '\n')
            # Need to convert: pred(...). -> pos(pred(...)).
            elif line.endswith(').') and not line.startswith('neg('):
                # Extract predicate
                match = re.match(r'(\w+)\((.*)\)\.', line)
                if match:
                    pred = match.group(1)
                    args = match.group(2)
                    f.write(f'pos({pred}({args})).\n')
            # Handle neg(pred(...)). format
            elif line.startswith('neg(') and line.endswith(').'):
                f.write(line + '\n')
    
    return True


def create_numsynth_bk_from_existing(bk_file, output_bk_file, problem_type='geometry'):
    """
    Convert BK file to NumSynth format.
    NumSynth needs arithmetic operations: add, mult, leq, geq
    """
    if bk_file is None or not os.path.exists(bk_file):
        # Create minimal BK with arithmetic
        with open(output_bk_file, 'w') as f:
            f.write("""% NumSynth-compatible background knowledge
% Arithmetic operations
precision(0.001).

my_add(A,B,C) :- nonvar(A), nonvar(B), C is A+B.
my_add(A,B,C) :- nonvar(A), nonvar(C), B is C-A.
my_add(A,B,C) :- nonvar(B), nonvar(C), A is C-B.

my_mult(A,B,C) :- nonvar(A), nonvar(B), nonvar(C), !, C1 is A*B, precision(P), C>=C1-P, C=<C1+P.
my_mult(A,B,C) :- nonvar(A), nonvar(B), !, C is A*B.
my_mult(A,B,C) :- nonvar(A), nonvar(C), !, nonzero(A), B is C/A.
my_mult(A,B,C) :- nonvar(B), nonvar(C), nonzero(B), A is C/B.

nonzero(A) :- nonvar(A), A>0.
nonzero(A) :- nonvar(A), A<0.

my_geq(A,B) :- nonvar(A), nonvar(B), A>=B.
my_leq(A,B) :- nonvar(A), nonvar(B), A=<B.
my_eq(A,A) :- nonvar(A).

% NumSynth numerical predicates
add(A,B,C) :- my_add(A,B,C).
mult(A,B,C) :- my_mult(A,B,C).
geq(A,B) :- my_geq(A,B).
leq(A,B) :- my_leq(A,B).
""")
        return True
    
    # Read existing BK and adapt
    with open(bk_file, 'r') as f:
        bk_content = f.read()
    
    with open(output_bk_file, 'w') as f:
        # Add NumSynth arithmetic if not present
        if 'add(' not in bk_content and 'mult(' not in bk_content:
            f.write("""% NumSynth-compatible background knowledge
precision(0.001).

my_add(A,B,C) :- nonvar(A), nonvar(B), C is A+B.
my_add(A,B,C) :- nonvar(A), nonvar(C), B is C-A.
my_add(A,B,C) :- nonvar(B), nonvar(C), A is C-B.

my_mult(A,B,C) :- nonvar(A), nonvar(B), nonvar(C), !, C1 is A*B, precision(P), C>=C1-P, C=<C1+P.
my_mult(A,B,C) :- nonvar(A), nonvar(B), !, C is A*B.
my_mult(A,B,C) :- nonvar(A), nonvar(C), !, nonzero(A), B is C/A.
my_mult(A,B,C) :- nonvar(B), nonvar(C), nonzero(B), A is C/B.

nonzero(A) :- nonvar(A), A>0.
nonzero(A) :- nonvar(A), A<0.

my_geq(A,B) :- nonvar(A), nonvar(B), A>=B.
my_leq(A,B) :- nonvar(A), nonvar(B), A=<B.
my_eq(A,A) :- nonvar(A).

add(A,B,C) :- my_add(A,B,C).
mult(A,B,C) :- my_mult(A,B,C).
geq(A,B) :- my_geq(A,B).
leq(A,B) :- my_leq(A,B).

""")
        
        # Add existing BK content (filter out conflicting definitions)
        for line in bk_content.split('\n'):
            if line.strip() and not line.strip().startswith('%'):
                # Skip if it's a conflicting arithmetic definition
                if any(op in line for op in ['add(', 'mult(', 'geq(', 'leq(']):
                    if not line.strip().startswith('my_'):
                        continue
            f.write(line + '\n')
    
    return True


def create_numsynth_bias(predicate_name, arity, types, output_file, max_numeric=3, max_vars=6, max_body=3):
    """
    Create NumSynth bias file for a predicate.
    
    Args:
        predicate_name: Name of target predicate
        arity: Number of arguments
        types: List of types for each argument (e.g., ['int', 'int'] or ['int', 'int', 'int'])
        output_file: Output bias file path
        max_numeric: Maximum number of numerical predicates
        max_vars: Maximum variables
        max_body: Maximum body literals
    """
    with open(output_file, 'w') as f:
        f.write(f"head_pred({predicate_name},{arity}).\n")
        type_str = ','.join(types)
        f.write(f"type({predicate_name},({type_str},)).\n")
        
        # Direction: all inputs
        dir_str = ','.join(['in'] * arity)
        f.write(f"direction({predicate_name},({dir_str},)).\n\n")
        
        f.write(f"max_vars({max_vars}).\n")
        f.write(f"max_body({max_body}).\n\n")
        
        f.write(f"max_numeric({max_numeric}).\n\n")
        
        # Add NumSynth numerical predicates
        f.write(f"numerical_pred(geq,2).\n")
        f.write(f"type(geq,(int,int)).\n")
        f.write(f"direction(geq,(in, out)).\n\n")
        
        f.write(f"numerical_pred(leq,2).\n")
        f.write(f"type(leq,(int,int)).\n")
        f.write(f"direction(leq,(in, out)).\n\n")
        
        f.write(f"numerical_pred(add,3).\n")
        f.write(f"type(add,(int, int, int)).\n")
        f.write(f"direction(add,(in,in,out)).\n\n")
        
        f.write(f"numerical_pred(mult,3).\n")
        f.write(f"type(mult,(int, int, int)).\n")
        f.write(f"direction(mult,(in,out,out)).\n\n")
        
        f.write(f"bounds(geq,1,(-100,100)).\n")
        f.write(f"bounds(leq,1,(-100,100)).\n")
        f.write(f"bounds(mult,1,(-100,100)).\n")
        f.write(f"bounds(add,1,(-100,100)).\n")


def convert_geometry0():
    """Convert geometry0 datasets (interval, halfplane)"""
    print("\n[Geometry0] Converting interval and halfplane...")
    
    geo0_dir = os.path.join(_experiments_dir, 'geometry0', 'data')
    output_base = os.path.join(_output_dir, 'geometry0')
    
    problems = [
        ('interval', 1, ['int']),
        ('halfplane', 2, ['int', 'int'])
    ]
    
    for prob_name, arity, types in problems:
        prob_dir = os.path.join(output_base, prob_name)
        ensure_dir(prob_dir)
        
        # Convert examples
        input_ex = os.path.join(geo0_dir, f'{prob_name}_examples.pl')
        output_ex = os.path.join(prob_dir, 'exs.pl')
        if convert_examples_file(input_ex, output_ex):
            print(f"  [OK] {prob_name}: converted examples")
        
        # Create BK (geometry0 doesn't have separate BK, learner creates it)
        output_bk = os.path.join(prob_dir, 'bk.pl')
        create_numsynth_bk_from_existing(None, output_bk)
        print(f"  [OK] {prob_name}: created BK")
        
        # Create bias
        output_bias = os.path.join(prob_dir, 'numsynth-bias.pl')
        create_numsynth_bias(prob_name, arity, types, output_bias, max_numeric=2 if prob_name == 'interval' else 3)
        print(f"  [OK] {prob_name}: created bias")


def convert_geometry1():
    """Convert geometry1 datasets"""
    print("\n[Geometry1] Converting 4 problems...")
    
    geo1_dir = os.path.join(_experiments_dir, 'geometry1', 'data')
    output_base = os.path.join(_output_dir, 'geometry1')
    
    problems = [
        ('halfplane3d', 3, ['int', 'int', 'int']),
        ('conjunction', 3, ['int', 'int', 'int']),
        ('multihalfplane', 2, ['int', 'int']),  # Fixed: multihalfplane is 2D (x, y), not 3D
        ('interval3d', 3, ['int', 'int', 'int'])
    ]
    
    for prob_name, arity, types in problems:
        prob_dir = os.path.join(output_base, prob_name)
        ensure_dir(prob_dir)
        
        # Convert examples
        input_ex = os.path.join(geo1_dir, f'{prob_name}_examples.pl')
        output_ex = os.path.join(prob_dir, 'exs.pl')
        if convert_examples_file(input_ex, output_ex):
            print(f"  [OK] {prob_name}: converted examples")
        
        # Convert BK
        input_bk = os.path.join(geo1_dir, f'{prob_name}_BK.pl')
        output_bk = os.path.join(prob_dir, 'bk.pl')
        create_numsynth_bk_from_existing(input_bk, output_bk)
        print(f"  [OK] {prob_name}: converted BK")
        
        # Create bias
        output_bias = os.path.join(prob_dir, 'numsynth-bias.pl')
        create_numsynth_bias(prob_name, arity, types, output_bias, max_numeric=4, max_vars=8, max_body=5)
        print(f"  [OK] {prob_name}: created bias")


def convert_geometry2():
    """Convert geometry2 datasets (relational)"""
    print("\n[Geometry2] Converting relational problems...")
    
    geo2_dir = os.path.join(_experiments_dir, 'geometry2', 'data')
    output_base = os.path.join(_output_dir, 'geometry2')
    
    problems = [
        ('left_of', 2, ['int', 'int']),
        ('inside', 2, ['int', 'int']),
        ('touching', 2, ['int', 'int']),
        ('overlapping', 2, ['int', 'int']),
        ('between', 3, ['int', 'int', 'int']),
        ('aligned', 3, ['int', 'int', 'int']),
        ('closer_than', 3, ['int', 'int', 'int']),
        ('near_corner', 2, ['int', 'int']),
        ('adjacent', 2, ['int', 'int']),
        ('surrounds', 2, ['int', 'int'])
    ]
    
    for prob_name, arity, types in problems:
        prob_dir = os.path.join(output_base, prob_name)
        ensure_dir(prob_dir)
        
        # Convert examples
        input_ex = os.path.join(geo2_dir, f'{prob_name}_examples.pl')
        output_ex = os.path.join(prob_dir, 'exs.pl')
        if convert_examples_file(input_ex, output_ex):
            print(f"  [OK] {prob_name}: converted examples")
        
        # Convert BK
        input_bk = os.path.join(geo2_dir, f'{prob_name}_BK.pl')
        output_bk = os.path.join(prob_dir, 'bk.pl')
        create_numsynth_bk_from_existing(input_bk, output_bk)
        print(f"  [OK] {prob_name}: converted BK")
        
        # Create bias (relational problems may need different settings)
        output_bias = os.path.join(prob_dir, 'numsynth-bias.pl')
        create_numsynth_bias(prob_name, arity, types, output_bias, max_numeric=2, max_vars=6, max_body=4)
        print(f"  [OK] {prob_name}: created bias")


def convert_geometry3():
    """Convert geometry3 datasets (nonlinear)"""
    print("\n[Geometry3] Converting nonlinear problems...")
    
    geo3_dir = os.path.join(_experiments_dir, 'geometry3', 'data')
    output_base = os.path.join(_output_dir, 'geometry3')
    
    # All geometry3 problems are 2D (x, y)
    problems = [
        ('in_circle', 2, ['int', 'int']),
        ('in_ellipse', 2, ['int', 'int']),
        ('hyperbola_side', 2, ['int', 'int']),
        ('xy_less_than', 2, ['int', 'int']),
        ('quad_strip', 2, ['int', 'int']),
        ('union_halfplanes', 2, ['int', 'int']),
        ('circle_or_box', 2, ['int', 'int']),
        ('piecewise', 2, ['int', 'int']),
        ('fallback_region', 2, ['int', 'int']),
        ('donut', 2, ['int', 'int']),
        ('lshape', 2, ['int', 'int']),
        ('above_parabola', 2, ['int', 'int']),
        ('sinusoidal', 2, ['int', 'int']),
        ('crescent', 2, ['int', 'int'])
    ]
    
    for prob_name, arity, types in problems:
        prob_dir = os.path.join(output_base, prob_name)
        ensure_dir(prob_dir)
        
        # Convert examples
        input_ex = os.path.join(geo3_dir, f'{prob_name}_examples.pl')
        output_ex = os.path.join(prob_dir, 'exs.pl')
        if convert_examples_file(input_ex, output_ex):
            print(f"  [OK] {prob_name}: converted examples")
        
        # Convert BK
        input_bk = os.path.join(geo3_dir, f'{prob_name}_BK.pl')
        output_bk = os.path.join(prob_dir, 'bk.pl')
        create_numsynth_bk_from_existing(input_bk, output_bk)
        print(f"  [OK] {prob_name}: converted BK")
        
        # Create bias (nonlinear needs more numeric predicates)
        output_bias = os.path.join(prob_dir, 'numsynth-bias.pl')
        create_numsynth_bias(prob_name, arity, types, output_bias, max_numeric=4, max_vars=8, max_body=5)
        print(f"  [OK] {prob_name}: created bias")


def convert_ip():
    """Convert IP datasets"""
    print("\n[IP] Converting InfluencePropagation tasks...")
    
    ip_dir = os.path.join(_experiments_dir, 'ip', 'data')
    output_base = os.path.join(_output_dir, 'ip')
    
    tasks = [
        ('ip1_active', 1, ['atom']),  # IP uses objects, but NumSynth may need different handling
        ('ip2_active', 1, ['atom']),
        ('ip3_active', 1, ['atom']),
        ('ip3_threshold', 1, ['atom']),
        ('ip4_high_score', 1, ['atom'])
    ]
    
    for task_name, arity, types in tasks:
        prob_dir = os.path.join(output_base, task_name)
        ensure_dir(prob_dir)
        
        # Convert examples
        input_ex = os.path.join(ip_dir, f'{task_name}_examples.pl')
        output_ex = os.path.join(prob_dir, 'exs.pl')
        if convert_examples_file(input_ex, output_ex):
            print(f"  [OK] {task_name}: converted examples")
        
        # Convert BK (IP uses objects_BK.pl)
        input_bk = os.path.join(ip_dir, 'objects_BK.pl')
        output_bk = os.path.join(prob_dir, 'bk.pl')
        create_numsynth_bk_from_existing(input_bk, output_bk)
        print(f"  [OK] {task_name}: converted BK")
        
        # Create bias (IP is relational, may need special handling)
        output_bias = os.path.join(prob_dir, 'numsynth-bias.pl')
        # IP uses object IDs, so we use 'atom' type
        create_numsynth_bias(task_name, arity, types, output_bias, max_numeric=3, max_vars=6, max_body=4)
        print(f"  [OK] {task_name}: created bias")


def main():
    """Main conversion function"""
    print("Converting all datasets to NumSynth format")
    print(f"\nOutput directory: {_output_dir}")
    
    # Create output directory
    ensure_dir(_output_dir)
    
    # Convert each dataset
    convert_geometry0()
    convert_geometry1()
    convert_geometry2()
    convert_geometry3()
    convert_ip()
    
    print("\nConversion complete!")
    print(f"\nAll NumSynth-compatible datasets are in: {_output_dir}")
    print("\nStructure:")
    print("  numsynth_datasets/")
    print("    geometry0/")
    print("      interval/")
    print("      halfplane/")
    print("    geometry1/")
    print("      halfplane3d/")
    print("      conjunction/")
    print("      multihalfplane/")
    print("      interval3d/")
    print("    geometry2/")
    print("      [10 relational problems]/")
    print("    geometry3/")
    print("      [7 nonlinear problems]/")
    print("    ip/")
    print("      [5 IP tasks]/")
    print("\nEach problem folder contains:")
    print("  - exs.pl (examples in pos/neg format)")
    print("  - bk.pl (background knowledge)")
    print("  - numsynth-bias.pl (bias file)")


if __name__ == "__main__":
    main()

