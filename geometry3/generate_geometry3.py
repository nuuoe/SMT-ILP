#!/usr/bin/env python3
"""
Generate Geometry3 problems: Nonlinear & Disjunctive Constraints

Category 1: Nonlinear Regions
1. Circle membership: x² + y² < r²
2. Elliptical region: (x-h)²/a² + (y-k)²/b² < 1
3. Hyperbolic region: x² - y² > k
4. Multiplicative constraint: x·y < c
5. Quadratic strips: a < x² < b

Category 2: Disjunctive (OR) Regions
6. Union of half-planes: (ax + by ≤ c) OR (dx + ey ≥ f)
7. Circle OR Box: (x² + y² < r²) OR (|x| < 2 AND |y| < 2)
8. Piecewise rule: (x < 0 AND y > 5) OR (x > 10)
9. Region with fallback: in_circle OR below_line

Category 3: Non-convex / Piecewise / Hybrid Regions
10. Donut (Annulus): r₁² < x² + y² < r₂²
11. L-shaped region: Union of two rectangles
12. Parabolic boundary: y > ax² + bx + c
13. Sinusoidal region: y > sin(x)
14. Crescent shape: (x² + y² < r₁²) AND ((x-h)² + (y-k)² > r₂²)
"""

import random
import math
import os

WORLD_SIZE = 100
PRECISION = 0.1


def gen_point_2d():
    """Generate a random 2D point"""
    return [random.randint(-WORLD_SIZE, WORLD_SIZE) for _ in range(2)]


# ========== Category 1: Nonlinear Regions ==========

def generate_circle(num_pos=30, num_neg=30):
    """Circle membership: x² + y² < r²"""
    r = random.randint(20, WORLD_SIZE//2)
    
    pos_examples = []
    neg_examples = []
    
    while len(pos_examples) < num_pos:
        [x, y] = gen_point_2d()
        if x*x + y*y < r*r - PRECISION:
            pos_examples.append(f'in_circle({x},{y},{r})')
    
    while len(neg_examples) < num_neg:
        [x, y] = gen_point_2d()
        if x*x + y*y >= r*r + PRECISION:
            neg_examples.append(f'in_circle({x},{y},{r})')
    
    return pos_examples, neg_examples, {'r': r}


def generate_ellipse(num_pos=30, num_neg=30):
    """Elliptical region: (x-h)²/a² + (y-k)²/b² < 1"""
    h = random.randint(-WORLD_SIZE//4, WORLD_SIZE//4)
    k = random.randint(-WORLD_SIZE//4, WORLD_SIZE//4)
    a = random.randint(15, WORLD_SIZE//3)
    b = random.randint(15, WORLD_SIZE//3)
    
    pos_examples = []
    neg_examples = []
    
    while len(pos_examples) < num_pos:
        [x, y] = gen_point_2d()
        dx, dy = x - h, y - k
        if (dx*dx)/(a*a) + (dy*dy)/(b*b) < 1 - PRECISION:
            pos_examples.append(f'in_ellipse({x},{y},{h},{k},{a},{b})')
    
    while len(neg_examples) < num_neg:
        [x, y] = gen_point_2d()
        dx, dy = x - h, y - k
        if (dx*dx)/(a*a) + (dy*dy)/(b*b) >= 1 + PRECISION:
            neg_examples.append(f'in_ellipse({x},{y},{h},{k},{a},{b})')
    
    return pos_examples, neg_examples, {'h': h, 'k': k, 'a': a, 'b': b}


def generate_hyperbola(num_pos=30, num_neg=30):
    """Hyperbolic region: x² - y² > k"""
    k = random.randint(-WORLD_SIZE//2, WORLD_SIZE//4)
    
    pos_examples = []
    neg_examples = []
    
    while len(pos_examples) < num_pos:
        [x, y] = gen_point_2d()
        if x*x - y*y > k + PRECISION:
            pos_examples.append(f'hyperbola_side({x},{y},{k})')
    
    while len(neg_examples) < num_neg:
        [x, y] = gen_point_2d()
        if x*x - y*y <= k - PRECISION:
            neg_examples.append(f'hyperbola_side({x},{y},{k})')
    
    return pos_examples, neg_examples, {'k': k}


def generate_xy_less_than(num_pos=30, num_neg=30):
    """Multiplicative constraint: x·y < c"""
    c = random.randint(-WORLD_SIZE//2, WORLD_SIZE//2)
    
    pos_examples = []
    neg_examples = []
    
    while len(pos_examples) < num_pos:
        [x, y] = gen_point_2d()
        if x * y < c - PRECISION:
            pos_examples.append(f'xy_less_than({x},{y},{c})')
    
    while len(neg_examples) < num_neg:
        [x, y] = gen_point_2d()
        if x * y >= c + PRECISION:
            neg_examples.append(f'xy_less_than({x},{y},{c})')
    
    return pos_examples, neg_examples, {'c': c}


def generate_quad_strip(num_pos=30, num_neg=30):
    """Quadratic strips: a < x² < b"""
    a = random.randint(0, WORLD_SIZE//4)
    b = random.randint(a + 10, WORLD_SIZE//2)
    
    pos_examples = []
    neg_examples = []
    
    while len(pos_examples) < num_pos:
        [x, y] = gen_point_2d()
        x_sq = x * x
        if a + PRECISION < x_sq < b - PRECISION:
            pos_examples.append(f'quad_strip({x},{y},{a},{b})')
    
    while len(neg_examples) < num_neg:
        [x, y] = gen_point_2d()
        x_sq = x * x
        if not (a + PRECISION < x_sq < b - PRECISION):
            neg_examples.append(f'quad_strip({x},{y},{a},{b})')
    
    return pos_examples, neg_examples, {'a': a, 'b': b}


# ========== Category 2: Disjunctive (OR) Regions ==========

def generate_union_halfplanes(num_pos=30, num_neg=30):
    """Union of half-planes: (ax + by ≤ c) OR (dx + ey ≥ f)"""
    # First halfplane: ax + by <= c
    a1 = random.randint(1, WORLD_SIZE//4)
    b1 = random.randint(1, WORLD_SIZE//4)
    c1 = random.randint(WORLD_SIZE//4, WORLD_SIZE//2)
    
    # Second halfplane: dx + ey >= f
    d2 = random.randint(1, WORLD_SIZE//4)
    e2 = random.randint(1, WORLD_SIZE//4)
    f2 = random.randint(-WORLD_SIZE//2, -WORLD_SIZE//4)
    
    pos_examples = []
    neg_examples = []
    
    while len(pos_examples) < num_pos:
        [x, y] = gen_point_2d()
        # Positive if EITHER condition is true
        cond1 = a1*x + b1*y <= c1 - PRECISION
        cond2 = d2*x + e2*y >= f2 + PRECISION
        if cond1 or cond2:
            pos_examples.append(f'union_halfplanes({x},{y})')
    
    while len(neg_examples) < num_neg:
        [x, y] = gen_point_2d()
        # Negative if NEITHER condition is true
        cond1 = a1*x + b1*y <= c1 - PRECISION
        cond2 = d2*x + e2*y >= f2 + PRECISION
        if not (cond1 or cond2):
            neg_examples.append(f'union_halfplanes({x},{y})')
    
    return pos_examples, neg_examples, {
        'a1': a1, 'b1': b1, 'c1': c1,
        'd2': d2, 'e2': e2, 'f2': f2
    }


def generate_circle_or_box(num_pos=30, num_neg=30):
    """Circle OR Box: (x² + y² < r²) OR (|x| < 2 AND |y| < 2)"""
    r = random.randint(30, WORLD_SIZE//2)
    box_size = random.randint(5, 15)
    
    pos_examples = []
    neg_examples = []
    
    while len(pos_examples) < num_pos:
        [x, y] = gen_point_2d()
        in_circle = x*x + y*y < r*r - PRECISION
        in_box = abs(x) < box_size - PRECISION and abs(y) < box_size - PRECISION
        if in_circle or in_box:
            pos_examples.append(f'circle_or_box({x},{y},{r},{box_size})')
    
    while len(neg_examples) < num_neg:
        [x, y] = gen_point_2d()
        in_circle = x*x + y*y < r*r - PRECISION
        in_box = abs(x) < box_size - PRECISION and abs(y) < box_size - PRECISION
        if not (in_circle or in_box):
            neg_examples.append(f'circle_or_box({x},{y},{r},{box_size})')
    
    return pos_examples, neg_examples, {'r': r, 'box_size': box_size}


def generate_piecewise(num_pos=30, num_neg=30):
    """Piecewise rule: (x < 0 AND y > 5) OR (x > 10)"""
    threshold_x1 = random.randint(-WORLD_SIZE//4, 0)
    threshold_y = random.randint(0, WORLD_SIZE//4)
    threshold_x2 = random.randint(10, WORLD_SIZE//4)
    
    pos_examples = []
    neg_examples = []
    
    while len(pos_examples) < num_pos:
        [x, y] = gen_point_2d()
        cond1 = x < threshold_x1 - PRECISION and y > threshold_y + PRECISION
        cond2 = x > threshold_x2 + PRECISION
        if cond1 or cond2:
            pos_examples.append(f'piecewise({x},{y})')
    
    while len(neg_examples) < num_neg:
        [x, y] = gen_point_2d()
        cond1 = x < threshold_x1 - PRECISION and y > threshold_y + PRECISION
        cond2 = x > threshold_x2 + PRECISION
        if not (cond1 or cond2):
            neg_examples.append(f'piecewise({x},{y})')
    
    return pos_examples, neg_examples, {
        'threshold_x1': threshold_x1, 'threshold_y': threshold_y, 'threshold_x2': threshold_x2
    }


def generate_fallback_region(num_pos=30, num_neg=30):
    """Region with fallback: in_circle OR below_line"""
    r = random.randint(20, WORLD_SIZE//2)
    # Line: ax + by + c = 0, below means ax + by + c < 0
    a = random.randint(1, WORLD_SIZE//4)
    b = random.randint(1, WORLD_SIZE//4)
    c = random.randint(-WORLD_SIZE//2, WORLD_SIZE//4)
    
    pos_examples = []
    neg_examples = []
    
    while len(pos_examples) < num_pos:
        [x, y] = gen_point_2d()
        in_circle = x*x + y*y < r*r - PRECISION
        below_line = a*x + b*y + c < -PRECISION
        if in_circle or below_line:
            pos_examples.append(f'fallback_region({x},{y},{r},{a},{b},{c})')
    
    while len(neg_examples) < num_neg:
        [x, y] = gen_point_2d()
        in_circle = x*x + y*y < r*r - PRECISION
        below_line = a*x + b*y + c < -PRECISION
        if not (in_circle or below_line):
            neg_examples.append(f'fallback_region({x},{y},{r},{a},{b},{c})')
    
    return pos_examples, neg_examples, {'r': r, 'a': a, 'b': b, 'c': c}


# ========== Category 3: Non-convex / Piecewise / Hybrid Regions ==========

def generate_donut(num_pos=30, num_neg=30):
    """Donut (Annulus): r₁² < x² + y² < r₂²"""
    r1 = random.randint(10, WORLD_SIZE//4)
    r2 = random.randint(r1 + 10, WORLD_SIZE//2)
    
    pos_examples = []
    neg_examples = []
    
    while len(pos_examples) < num_pos:
        [x, y] = gen_point_2d()
        dist_sq = x*x + y*y
        if r1*r1 + PRECISION < dist_sq < r2*r2 - PRECISION:
            pos_examples.append(f'donut({x},{y},{r1},{r2})')
    
    while len(neg_examples) < num_neg:
        [x, y] = gen_point_2d()
        dist_sq = x*x + y*y
        if not (r1*r1 + PRECISION < dist_sq < r2*r2 - PRECISION):
            neg_examples.append(f'donut({x},{y},{r1},{r2})')
    
    return pos_examples, neg_examples, {'r1': r1, 'r2': r2}


def generate_lshape(num_pos=30, num_neg=30):
    """L-shaped region: Union of two rectangles"""
    # Rectangle 1: x_min1 < x < x_max1 AND y_min1 < y < y_max1
    x_min1 = random.randint(-WORLD_SIZE//2, 0)
    x_max1 = random.randint(0, WORLD_SIZE//4)
    y_min1 = random.randint(-WORLD_SIZE//2, WORLD_SIZE//4)
    y_max1 = random.randint(y_min1 + 10, WORLD_SIZE//2)
    
    # Rectangle 2: x_min2 < x < x_max2 AND y_min2 < y < y_max2
    x_min2 = random.randint(x_max1 - 5, x_max1 + 5)  # Overlap with rect1
    x_max2 = random.randint(x_max1 + 10, WORLD_SIZE//2)
    y_min2 = random.randint(-WORLD_SIZE//2, y_max1 - 5)  # Below rect1
    y_max2 = random.randint(y_min2 + 10, y_max1 + 5)  # Overlap with rect1
    
    pos_examples = []
    neg_examples = []
    
    while len(pos_examples) < num_pos:
        [x, y] = gen_point_2d()
        in_rect1 = x_min1 < x < x_max1 and y_min1 < y < y_max1
        in_rect2 = x_min2 < x < x_max2 and y_min2 < y < y_max2
        if in_rect1 or in_rect2:
            pos_examples.append(f'lshape({x},{y})')
    
    while len(neg_examples) < num_neg:
        [x, y] = gen_point_2d()
        in_rect1 = x_min1 < x < x_max1 and y_min1 < y < y_max1
        in_rect2 = x_min2 < x < x_max2 and y_min2 < y < y_max2
        if not (in_rect1 or in_rect2):
            neg_examples.append(f'lshape({x},{y})')
    
    return pos_examples, neg_examples, {
        'x_min1': x_min1, 'x_max1': x_max1, 'y_min1': y_min1, 'y_max1': y_max1,
        'x_min2': x_min2, 'x_max2': x_max2, 'y_min2': y_min2, 'y_max2': y_max2
    }


def generate_parabola(num_pos=30, num_neg=30):
    """Parabolic boundary: y > ax² + bx + c"""
    a = random.uniform(0.01, 0.1)  # Small positive coefficient
    b = random.randint(-WORLD_SIZE//4, WORLD_SIZE//4)
    c = random.randint(-WORLD_SIZE//2, WORLD_SIZE//4)
    
    pos_examples = []
    neg_examples = []
    
    while len(pos_examples) < num_pos:
        [x, y] = gen_point_2d()
        parabola_val = a*x*x + b*x + c
        if y > parabola_val + PRECISION:
            pos_examples.append(f'above_parabola({x},{y},{a:.4f},{b},{c})')
    
    while len(neg_examples) < num_neg:
        [x, y] = gen_point_2d()
        parabola_val = a*x*x + b*x + c
        if y <= parabola_val - PRECISION:
            neg_examples.append(f'above_parabola({x},{y},{a:.4f},{b},{c})')
    
    return pos_examples, neg_examples, {'a': a, 'b': b, 'c': c}


def generate_sinusoidal(num_pos=30, num_neg=30):
    """Sinusoidal region: y > sin(x)"""
    # Scale x to reasonable range for sin
    scale = random.uniform(0.1, 0.5)
    offset = random.uniform(-2, 2)
    
    pos_examples = []
    neg_examples = []
    
    while len(pos_examples) < num_pos:
        [x, y] = gen_point_2d()
        sin_val = math.sin(scale * x) + offset
        if y > sin_val + PRECISION:
            pos_examples.append(f'sinusoidal({x},{y},{scale:.4f},{offset:.4f})')
    
    while len(neg_examples) < num_neg:
        [x, y] = gen_point_2d()
        sin_val = math.sin(scale * x) + offset
        if y <= sin_val - PRECISION:
            neg_examples.append(f'sinusoidal({x},{y},{scale:.4f},{offset:.4f})')
    
    return pos_examples, neg_examples, {'scale': scale, 'offset': offset}


def generate_crescent(num_pos=30, num_neg=30):
    """Crescent shape: (x² + y² < r₁²) AND ((x-h)² + (y-k)² > r₂²)"""
    r1 = random.randint(20, WORLD_SIZE//2)
    h = random.randint(-WORLD_SIZE//4, WORLD_SIZE//4)
    k = random.randint(-WORLD_SIZE//4, WORLD_SIZE//4)
    r2 = random.randint(10, r1 - 5)  # Smaller than r1
    
    pos_examples = []
    neg_examples = []
    
    while len(pos_examples) < num_pos:
        [x, y] = gen_point_2d()
        in_circle1 = x*x + y*y < r1*r1 - PRECISION
        dx, dy = x - h, y - k
        out_circle2 = dx*dx + dy*dy > r2*r2 + PRECISION
        if in_circle1 and out_circle2:
            pos_examples.append(f'crescent({x},{y},{r1},{h},{k},{r2})')
    
    while len(neg_examples) < num_neg:
        [x, y] = gen_point_2d()
        in_circle1 = x*x + y*y < r1*r1 - PRECISION
        dx, dy = x - h, y - k
        out_circle2 = dx*dx + dy*dy > r2*r2 + PRECISION
        if not (in_circle1 and out_circle2):
            neg_examples.append(f'crescent({x},{y},{r1},{h},{k},{r2})')
    
    return pos_examples, neg_examples, {'r1': r1, 'h': h, 'k': k, 'r2': r2}


def write_examples_file(filepath, pos_examples, neg_examples):
    """Write examples to a Prolog file"""
    with open(filepath, 'w') as f:
        f.write("% Positive examples\n")
        for ex in pos_examples:
            f.write(f"{ex}.\n")
        f.write("\n% Negative examples\n")
        for ex in neg_examples:
            f.write(f"not({ex}).\n")


def write_bk_file(filepath, problem_type):
    """Write background knowledge file"""
    with open(filepath, 'w') as f:
        f.write("% Background knowledge for geometry3 problems\n")
        f.write("% This file is a template - the learner will add feature facts dynamically\n\n")
        f.write(f"% Problem type: {problem_type}\n")


if __name__ == "__main__":
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    os.makedirs(data_dir, exist_ok=True)
    
    print("Generating Geometry3 datasets...")
    
    # Category 1: Nonlinear Regions
    print("\nCategory 1: Nonlinear Regions")
    
    print("Generating in_circle problem...")
    pos, neg, params = generate_circle()
    write_examples_file(os.path.join(data_dir, 'in_circle_examples.pl'), pos, neg)
    write_bk_file(os.path.join(data_dir, 'in_circle_BK.pl'), 'in_circle')
    print(f"  Generated {len(pos)} positive and {len(neg)} negative examples")
    
    print("Generating in_ellipse problem...")
    pos, neg, params = generate_ellipse()
    write_examples_file(os.path.join(data_dir, 'in_ellipse_examples.pl'), pos, neg)
    write_bk_file(os.path.join(data_dir, 'in_ellipse_BK.pl'), 'in_ellipse')
    print(f"  Generated {len(pos)} positive and {len(neg)} negative examples")
    
    print("Generating hyperbola_side problem...")
    pos, neg, params = generate_hyperbola()
    write_examples_file(os.path.join(data_dir, 'hyperbola_side_examples.pl'), pos, neg)
    write_bk_file(os.path.join(data_dir, 'hyperbola_side_BK.pl'), 'hyperbola_side')
    print(f"  Generated {len(pos)} positive and {len(neg)} negative examples")
    
    print("Generating xy_less_than problem...")
    pos, neg, params = generate_xy_less_than()
    write_examples_file(os.path.join(data_dir, 'xy_less_than_examples.pl'), pos, neg)
    write_bk_file(os.path.join(data_dir, 'xy_less_than_BK.pl'), 'xy_less_than')
    print(f"  Generated {len(pos)} positive and {len(neg)} negative examples")
    
    print("Generating quad_strip problem...")
    pos, neg, params = generate_quad_strip()
    write_examples_file(os.path.join(data_dir, 'quad_strip_examples.pl'), pos, neg)
    write_bk_file(os.path.join(data_dir, 'quad_strip_BK.pl'), 'quad_strip')
    print(f"  Generated {len(pos)} positive and {len(neg)} negative examples")
    
    # Category 2: Disjunctive (OR) Regions
    print("\nCategory 2: Disjunctive (OR) Regions")
    
    print("Generating union_halfplanes problem...")
    pos, neg, params = generate_union_halfplanes()
    write_examples_file(os.path.join(data_dir, 'union_halfplanes_examples.pl'), pos, neg)
    write_bk_file(os.path.join(data_dir, 'union_halfplanes_BK.pl'), 'union_halfplanes')
    print(f"  Generated {len(pos)} positive and {len(neg)} negative examples")
    
    print("Generating circle_or_box problem...")
    pos, neg, params = generate_circle_or_box()
    write_examples_file(os.path.join(data_dir, 'circle_or_box_examples.pl'), pos, neg)
    write_bk_file(os.path.join(data_dir, 'circle_or_box_BK.pl'), 'circle_or_box')
    print(f"  Generated {len(pos)} positive and {len(neg)} negative examples")
    
    print("Generating piecewise problem...")
    pos, neg, params = generate_piecewise()
    write_examples_file(os.path.join(data_dir, 'piecewise_examples.pl'), pos, neg)
    write_bk_file(os.path.join(data_dir, 'piecewise_BK.pl'), 'piecewise')
    print(f"  Generated {len(pos)} positive and {len(neg)} negative examples")
    
    print("Generating fallback_region problem...")
    pos, neg, params = generate_fallback_region()
    write_examples_file(os.path.join(data_dir, 'fallback_region_examples.pl'), pos, neg)
    write_bk_file(os.path.join(data_dir, 'fallback_region_BK.pl'), 'fallback_region')
    print(f"  Generated {len(pos)} positive and {len(neg)} negative examples")
    
    # Category 3: Non-convex / Piecewise / Hybrid Regions
    print("\nCategory 3: Non-convex / Piecewise / Hybrid Regions")
    
    print("Generating donut problem...")
    pos, neg, params = generate_donut()
    write_examples_file(os.path.join(data_dir, 'donut_examples.pl'), pos, neg)
    write_bk_file(os.path.join(data_dir, 'donut_BK.pl'), 'donut')
    print(f"  Generated {len(pos)} positive and {len(neg)} negative examples")
    
    print("Generating lshape problem...")
    pos, neg, params = generate_lshape()
    write_examples_file(os.path.join(data_dir, 'lshape_examples.pl'), pos, neg)
    write_bk_file(os.path.join(data_dir, 'lshape_BK.pl'), 'lshape')
    print(f"  Generated {len(pos)} positive and {len(neg)} negative examples")
    
    print("Generating above_parabola problem...")
    pos, neg, params = generate_parabola()
    write_examples_file(os.path.join(data_dir, 'above_parabola_examples.pl'), pos, neg)
    write_bk_file(os.path.join(data_dir, 'above_parabola_BK.pl'), 'above_parabola')
    print(f"  Generated {len(pos)} positive and {len(neg)} negative examples")
    
    print("Generating sinusoidal problem...")
    pos, neg, params = generate_sinusoidal()
    write_examples_file(os.path.join(data_dir, 'sinusoidal_examples.pl'), pos, neg)
    write_bk_file(os.path.join(data_dir, 'sinusoidal_BK.pl'), 'sinusoidal')
    print(f"  Generated {len(pos)} positive and {len(neg)} negative examples")
    
    print("Generating crescent problem...")
    pos, neg, params = generate_crescent()
    write_examples_file(os.path.join(data_dir, 'crescent_examples.pl'), pos, neg)
    write_bk_file(os.path.join(data_dir, 'crescent_BK.pl'), 'crescent')
    print(f"  Generated {len(pos)} positive and {len(neg)} negative examples")
    
    print("\nAll Geometry3 datasets generated successfully!")
    print(f"Total: 14 problems across 3 categories")
