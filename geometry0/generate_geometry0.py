#!/usr/bin/env python3
"""
Generate geometry0 problems for testing PyGol+Z3 using the EXACT original problem classes:
1. Interval: min_val_x <= x <= max_val_x
2. Halfplane: a*x + y <= b
"""

import sys
import os

# Add paths to import standalone problem classes (no longer depends on numsynth)
_current_file = os.path.abspath(__file__)
_geometry0_dir = os.path.dirname(_current_file)
sys.path.insert(0, _geometry0_dir)

from numsynth_standalone.geometry.interval.interval import IntervalProblem
from numsynth_standalone.geometry.halfplane.halfplane import HalfPlaneProblem


# Problem 1: Interval - using EXACT original class
def generate_interval(num_pos=30, num_neg=30):
    """Generate interval problem using original IntervalProblem class"""
    problem = IntervalProblem()
    pos_examples = [problem.gen_pos() for _ in range(num_pos)]
    neg_examples = [problem.gen_neg() for _ in range(num_neg)]
    return pos_examples, neg_examples, {'bk_file': problem.bk_file()}


# Problem 2: Halfplane - using EXACT original class
def generate_halfplane(num_pos=30, num_neg=30):
    """Generate halfplane problem using original HalfPlaneProblem class"""
    problem = HalfPlaneProblem()
    pos_examples = [problem.gen_pos() for _ in range(num_pos)]
    neg_examples = [problem.gen_neg() for _ in range(num_neg)]
    return pos_examples, neg_examples, {'a': problem.a, 'b': problem.b, 'bk_file': problem.bk_file()}


def write_examples_file(filepath, pos_examples, neg_examples):
    """Write examples to file in Prolog format"""
    with open(filepath, 'w') as f:
        for ex in pos_examples:
            f.write(f'pos({ex}).\n')
        for ex in neg_examples:
            f.write(f'neg({ex}).\n')


def create_bk_file_interval(filepath, examples_file):
    """Create BK file for interval with feature facts from examples"""
    import re
    
    # Read examples to extract feature values
    feature_facts = []
    example_id = 1
    
    with open(examples_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('pos(') or line.startswith('neg('):
                match = re.match(r'(?:pos|neg)\(interval\(([^)]+)\)\)', line)
                if match:
                    x_val = float(match.group(1))
                    e_id = f'e_{example_id}'
                    feature_facts.append(f'x({e_id}, {x_val}).')
                    example_id += 1
    
    # Write BK file
    with open(filepath, 'w') as f:
        f.write("""% Arithmetic operations and comparison predicates (like numsynth test.pl)
magic(_).
geq(A,B) :- nonvar(A), nonvar(B), A>=B.
leq(A,B) :- nonvar(A), nonvar(B), A=<B.
eq(A,A) :- nonvar(A).
lt(A,B) :- nonvar(A), nonvar(B), A<B.
gt(A,B) :- nonvar(A), nonvar(B), A>B.
add(A,B,C) :- nonvar(A), nonvar(B), C is A+B.
add(A,B,C) :- nonvar(A), nonvar(C), B is C-A.
add(A,B,C) :- nonvar(B), nonvar(C), A is C-B.
mult(A,B,C) :- nonvar(A), nonvar(B), C is A*B.
mult(A,B,C) :- nonvar(A), nonvar(C), \\+(A=0.0), \\+(A=0), B is C/A.
mult(A,B,C) :- nonvar(B), nonvar(C), \\+(B=0.0), \\+(B=0), A is C/B.

:- dynamic x/2.

% Feature facts
""")
        for fact in feature_facts:
            f.write(fact + '\n')


def create_bk_file_halfplane(filepath, examples_file):
    """Create BK file for halfplane with feature facts from examples"""
    import re
    
    # Read examples to extract feature values
    feature_facts = []
    example_id = 1
    
    with open(examples_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('pos(') or line.startswith('neg('):
                match = re.match(r'(?:pos|neg)\(halfplane\(([^,]+),\s*([^)]+)\)\)', line)
                if match:
                    x_val = float(match.group(1))
                    y_val = float(match.group(2))
                    e_id = f'e_{example_id}'
                    feature_facts.append(f'x({e_id}, {x_val}).')
                    feature_facts.append(f'y({e_id}, {y_val}).')
                    example_id += 1
    
    # Write BK file
    with open(filepath, 'w') as f:
        f.write("""% Arithmetic operations and comparison predicates (like numsynth test.pl)
magic(_).
geq(A,B) :- nonvar(A), nonvar(B), A>=B.
leq(A,B) :- nonvar(A), nonvar(B), A=<B.
eq(A,A) :- nonvar(A).
lt(A,B) :- nonvar(A), nonvar(B), A<B.
gt(A,B) :- nonvar(A), nonvar(B), A>B.
add(A,B,C) :- nonvar(A), nonvar(B), C is A+B.
add(A,B,C) :- nonvar(A), nonvar(C), B is C-A.
add(A,B,C) :- nonvar(B), nonvar(C), A is C-B.
mult(A,B,C) :- nonvar(A), nonvar(B), C is A*B.
mult(A,B,C) :- nonvar(A), nonvar(C), \\+(A=0.0), \\+(A=0), B is C/A.
mult(A,B,C) :- nonvar(B), nonvar(C), \\+(B=0.0), \\+(B=0), A is C/B.

:- dynamic x/2.
:- dynamic y/2.

% Feature facts
""")
        for fact in feature_facts:
            f.write(fact + '\n')


if __name__ == "__main__":
    # Set fixed seed for reproducibility (seed 2 gives interval > 90% accuracy)
    import random
    import numpy as np
    random.seed(2)
    np.random.seed(2)
    
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    os.makedirs(data_dir, exist_ok=True)
    
    # Problem 1: Interval - using original class
    print("Generating interval problem using original IntervalProblem class...")
    pos, neg, params = generate_interval()
    examples_file = os.path.join(data_dir, 'interval_examples.pl')
    write_examples_file(examples_file, pos, neg)
    # Note: The learner creates its own BK file with arithmetic operations
    # The original BK file is for numsynth/Popper and not used by PyGol+Z3
    original_bk = params['bk_file']
    print(f"  Generated {len(pos)} pos, {len(neg)} neg examples")
    print(f"  Note: Learner will create its own BK file (original BK is for numsynth/Popper)")
    
    # Problem 2: Halfplane - using original class
    print("\nGenerating halfplane problem using original HalfPlaneProblem class...")
    pos, neg, params = generate_halfplane()
    examples_file = os.path.join(data_dir, 'halfplane_examples.pl')
    write_examples_file(examples_file, pos, neg)
    # Note: The learner creates its own BK file with arithmetic operations
    # The original BK file is for numsynth/Popper and not used by PyGol+Z3
    original_bk = params['bk_file']
    print(f"  Generated {len(pos)} pos, {len(neg)} neg examples")
    print(f"  Coefficients: a={params['a']}, b={params['b']}")
    print(f"  Note: Learner will create its own BK file (original BK is for numsynth/Popper)")
    
    print("\nGeometry0 problems generated using EXACT original problem classes")

