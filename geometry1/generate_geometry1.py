#!/usr/bin/env python3
"""
Generate more difficult geometry problems for testing PyGol+Z3:
1. 3D Halfplane: a*x + b*y + c*z <= d
2. Conjunction: (x + y <= 5) AND (2 < z < 10)
3. Multiple Halfplanes: (a1*x + b1*y <= c1) AND (a2*x + b2*y <= c2)
4. 3D Interval: (x_min < x < x_max) AND (y_min < y < y_max) AND (z_min < z < z_max)
"""

import random
import os

WORLD_SIZE = 100
PRECISION = 0.1

def gen_point_3d():
    """Generate a random 3D point"""
    return [random.randint(-WORLD_SIZE, WORLD_SIZE) for _ in range(3)]


# Problem 1: 3D Halfplane
def generate_3d_halfplane(num_pos=30, num_neg=30):
    """Generate 3D halfplane problem: a*x + b*y + c*z <= d"""
    a = random.randint(1, WORLD_SIZE//4)
    b = random.randint(1, WORLD_SIZE//4)
    c = random.randint(1, WORLD_SIZE//4)
    d = random.randint(WORLD_SIZE//4, WORLD_SIZE//2)
    
    pos_examples = []
    neg_examples = []
    
    # Generate positive examples
    while len(pos_examples) < num_pos:
        [x, y, z] = gen_point_3d()
        if a*x + b*y + c*z <= d - PRECISION:
            pos_examples.append(f'halfplane3d({x},{y},{z})')
    
    # Generate negative examples
    while len(neg_examples) < num_neg:
        [x, y, z] = gen_point_3d()
        if a*x + b*y + c*z > d + PRECISION:
            neg_examples.append(f'halfplane3d({x},{y},{z})')
    
    return pos_examples, neg_examples, {'a': a, 'b': b, 'c': c, 'd': d}


# Problem 2: Conjunction (Halfplane + Interval)
def generate_conjunction(num_pos=30, num_neg=30):
    """Generate conjunction problem: (x + y <= threshold) AND (z_min < z < z_max)"""
    # Halfplane constraint: x + y <= threshold
    threshold = random.randint(WORLD_SIZE//4, WORLD_SIZE//2)
    
    # Interval constraint: z_min < z < z_max
    z_min = random.randint(-WORLD_SIZE//2, 0)
    z_max = random.randint(0, WORLD_SIZE//2)
    
    pos_examples = []
    neg_examples = []
    
    # Generate positive examples
    while len(pos_examples) < num_pos:
        [x, y, z] = gen_point_3d()
        if (x + y <= threshold - PRECISION) and (z_min < z < z_max):
            pos_examples.append(f'conjunction({x},{y},{z})')
    
    # Generate negative examples
    while len(neg_examples) < num_neg:
        [x, y, z] = gen_point_3d()
        if not ((x + y <= threshold - PRECISION) and (z_min < z < z_max)):
            neg_examples.append(f'conjunction({x},{y},{z})')
    
    return pos_examples, neg_examples, {'threshold': threshold, 'z_min': z_min, 'z_max': z_max}


# Problem 3: Multiple Halfplanes (Intersection)
def generate_multiple_halfplanes(num_pos=30, num_neg=30):
    """Generate multiple halfplanes: (a1*x + b1*y <= c1) AND (a2*x + b2*y <= c2)"""
    # First halfplane
    a1 = random.randint(1, WORLD_SIZE//4)
    b1 = random.randint(1, WORLD_SIZE//4)
    c1 = random.randint(WORLD_SIZE//4, WORLD_SIZE//2)
    
    # Second halfplane (different coefficients)
    a2 = random.randint(1, WORLD_SIZE//4)
    b2 = random.randint(1, WORLD_SIZE//4)
    c2 = random.randint(WORLD_SIZE//4, WORLD_SIZE//2)
    
    pos_examples = []
    neg_examples = []
    
    # Generate positive examples
    while len(pos_examples) < num_pos:
        [x, y] = [random.randint(-WORLD_SIZE, WORLD_SIZE) for _ in range(2)]
        if (a1*x + b1*y <= c1 - PRECISION) and (a2*x + b2*y <= c2 - PRECISION):
            pos_examples.append(f'multihalfplane({x},{y})')
    
    # Generate negative examples
    while len(neg_examples) < num_neg:
        [x, y] = [random.randint(-WORLD_SIZE, WORLD_SIZE) for _ in range(2)]
        if not ((a1*x + b1*y <= c1 - PRECISION) and (a2*x + b2*y <= c2 - PRECISION)):
            neg_examples.append(f'multihalfplane({x},{y})')
    
    return pos_examples, neg_examples, {
        'a1': a1, 'b1': b1, 'c1': c1,
        'a2': a2, 'b2': b2, 'c2': c2
    }


# Problem 4: 3D Interval
def generate_3d_interval(num_pos=30, num_neg=30):
    """Generate 3D interval: (x_min < x < x_max) AND (y_min < y < y_max) AND (z_min < z < z_max)"""
    x_min = random.randint(-WORLD_SIZE//2, 0)
    x_max = random.randint(0, WORLD_SIZE//2)
    y_min = random.randint(-WORLD_SIZE//2, 0)
    y_max = random.randint(0, WORLD_SIZE//2)
    z_min = random.randint(-WORLD_SIZE//2, 0)
    z_max = random.randint(0, WORLD_SIZE//2)
    
    pos_examples = []
    neg_examples = []
    
    # Generate positive examples
    while len(pos_examples) < num_pos:
        [x, y, z] = gen_point_3d()
        if (x_min < x < x_max) and (y_min < y < y_max) and (z_min < z < z_max):
            pos_examples.append(f'interval3d({x},{y},{z})')
    
    # Generate negative examples
    while len(neg_examples) < num_neg:
        [x, y, z] = gen_point_3d()
        if not ((x_min < x < x_max) and (y_min < y < y_max) and (z_min < z < z_max)):
            neg_examples.append(f'interval3d({x},{y},{z})')
    
    return pos_examples, neg_examples, {
        'x_min': x_min, 'x_max': x_max,
        'y_min': y_min, 'y_max': y_max,
        'z_min': z_min, 'z_max': z_max
    }


def write_examples_file(filepath, pos_examples, neg_examples):
    """Write examples to file in Prolog format"""
    with open(filepath, 'w') as f:
        for ex in pos_examples:
            f.write(f'pos({ex}).\n')
        for ex in neg_examples:
            f.write(f'neg({ex}).\n')


def create_bk_file_3d_halfplane(filepath, examples_file):
    """Create BK file for 3D halfplane with feature facts from examples"""
    import re
    
    # Read examples to extract feature values
    feature_facts = []
    example_id = 1
    
    with open(examples_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('pos(') or line.startswith('neg('):
                match = re.match(r'(?:pos|neg)\(halfplane3d\(([^,]+),\s*([^,]+),\s*([^)]+)\)\)', line)
                if match:
                    x_val = float(match.group(1))
                    y_val = float(match.group(2))
                    z_val = float(match.group(3))
                    e_id = f'e_{example_id}'
                    feature_facts.append(f'x({e_id}, {x_val}).')
                    feature_facts.append(f'y({e_id}, {y_val}).')
                    feature_facts.append(f'z({e_id}, {z_val}).')
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
:- dynamic z/2.

% Feature facts
""")
        for fact in feature_facts:
            f.write(fact + '\n')


def create_bk_file_conjunction(filepath, examples_file):
    """Create BK file for conjunction with feature facts from examples"""
    import re
    
    # Read examples to extract feature values
    feature_facts = []
    example_id = 1
    
    with open(examples_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('pos(') or line.startswith('neg('):
                match = re.match(r'(?:pos|neg)\(conjunction\(([^,]+),\s*([^,]+),\s*([^)]+)\)\)', line)
                if match:
                    x_val = float(match.group(1))
                    y_val = float(match.group(2))
                    z_val = float(match.group(3))
                    e_id = f'e_{example_id}'
                    feature_facts.append(f'x({e_id}, {x_val}).')
                    feature_facts.append(f'y({e_id}, {y_val}).')
                    feature_facts.append(f'z({e_id}, {z_val}).')
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
:- dynamic z/2.

% Feature facts
""")
        for fact in feature_facts:
            f.write(fact + '\n')


def create_bk_file_multihalfplane(filepath, examples_file):
    """Create BK file for multiple halfplanes with feature facts from examples"""
    import re
    
    # Read examples to extract feature values
    feature_facts = []
    example_id = 1
    
    with open(examples_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('pos(') or line.startswith('neg('):
                match = re.match(r'(?:pos|neg)\(multihalfplane\(([^,]+),\s*([^)]+)\)\)', line)
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


def create_bk_file_3d_interval(filepath, examples_file):
    """Create BK file for 3D interval with feature facts from examples"""
    import re
    
    # Read examples to extract feature values
    feature_facts = []
    example_id = 1
    
    with open(examples_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('pos(') or line.startswith('neg('):
                match = re.match(r'(?:pos|neg)\(interval3d\(([^,]+),\s*([^,]+),\s*([^)]+)\)\)', line)
                if match:
                    x_val = float(match.group(1))
                    y_val = float(match.group(2))
                    z_val = float(match.group(3))
                    e_id = f'e_{example_id}'
                    feature_facts.append(f'x({e_id}, {x_val}).')
                    feature_facts.append(f'y({e_id}, {y_val}).')
                    feature_facts.append(f'z({e_id}, {z_val}).')
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
:- dynamic z/2.

% Feature facts
""")
        for fact in feature_facts:
            f.write(fact + '\n')


if __name__ == "__main__":
    # Create data directory
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    os.makedirs(data_dir, exist_ok=True)
    
    # Problem 1: 3D Halfplane
    pos, neg, params = generate_3d_halfplane()
    examples_file = os.path.join(data_dir, 'halfplane3d_examples.pl')
    write_examples_file(examples_file, pos, neg)
    create_bk_file_3d_halfplane(os.path.join(data_dir, 'halfplane3d_BK.pl'), examples_file)
    
    # Problem 2: Conjunction
    pos, neg, params = generate_conjunction()
    examples_file = os.path.join(data_dir, 'conjunction_examples.pl')
    write_examples_file(examples_file, pos, neg)
    create_bk_file_conjunction(os.path.join(data_dir, 'conjunction_BK.pl'), examples_file)
    
    # Problem 3: Multiple Halfplanes
    pos, neg, params = generate_multiple_halfplanes()
    examples_file = os.path.join(data_dir, 'multihalfplane_examples.pl')
    write_examples_file(examples_file, pos, neg)
    create_bk_file_multihalfplane(os.path.join(data_dir, 'multihalfplane_BK.pl'), examples_file)
    
    # Problem 4: 3D Interval
    pos, neg, params = generate_3d_interval()
    examples_file = os.path.join(data_dir, 'interval3d_examples.pl')
    write_examples_file(examples_file, pos, neg)
    create_bk_file_3d_interval(os.path.join(data_dir, 'interval3d_BK.pl'), examples_file)
    
    print("Geometry1 problems generated")

