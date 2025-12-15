#!/usr/bin/env python3
"""
Load geometry3 problems from Prolog files
Geometry3: Nonlinear & Disjunctive Constraints
"""

import re
import pandas as pd
import numpy as np
import os
import math


def parse_point(point_str):
    """Parse point string 'x,y' into list of floats"""
    return [float(x.strip()) for x in point_str.split(',')]


# ========== Category 1: Nonlinear Regions ==========

def load_in_circle_data(examples_file, bk_file=None):
    """Load in_circle data: in_circle(x, y, r) where x² + y² < r²"""
    pos_examples = []
    neg_examples = []
    
    with open(examples_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('in_circle(') and not line.startswith('not('):
                match = re.match(r'in_circle\(([^,]+),([^,]+),([^)]+)\)', line.rstrip('.'))
                if match:
                    x, y, r = float(match.group(1)), float(match.group(2)), float(match.group(3))
                    pos_examples.append({
                        'x': x, 'y': y, 'r': r,
                        'x_sq': x*x, 'y_sq': y*y, 'dist_sq': x*x + y*y
                    })
            elif line.startswith('not(in_circle('):
                match = re.match(r'not\(in_circle\(([^,]+),([^,]+),([^)]+)\)\)', line.rstrip('.'))
                if match:
                    x, y, r = float(match.group(1)), float(match.group(2)), float(match.group(3))
                    neg_examples.append({
                        'x': x, 'y': y, 'r': r,
                        'x_sq': x*x, 'y_sq': y*y, 'dist_sq': x*x + y*y
                    })
    
    X_list = pos_examples + neg_examples
    y_list = [1] * len(pos_examples) + [0] * len(neg_examples)
    
    X = pd.DataFrame(X_list)
    y = np.array(y_list)
    X.index = [f"e_{i+1}" for i in range(len(X))]
    
    return X, y, bk_file


def load_in_ellipse_data(examples_file, bk_file=None):
    """Load in_ellipse data: (x-h)²/a² + (y-k)²/b² < 1"""
    pos_examples = []
    neg_examples = []
    
    with open(examples_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('in_ellipse(') and not line.startswith('not('):
                match = re.match(r'in_ellipse\(([^,]+),([^,]+),([^,]+),([^,]+),([^,]+),([^)]+)\)', line.rstrip('.'))
                if match:
                    x, y, h, k, a, b = [float(match.group(i)) for i in range(1, 7)]
                    dx, dy = x - h, y - k
                    pos_examples.append({
                        'x': x, 'y': y, 'h': h, 'k': k, 'a': a, 'b': b,
                        'dx': dx, 'dy': dy,
                        'dx_sq': dx*dx, 'dy_sq': dy*dy,
                        'ellipse_val': (dx*dx)/(a*a) + (dy*dy)/(b*b)
                    })
            elif line.startswith('not(in_ellipse('):
                match = re.match(r'not\(in_ellipse\(([^,]+),([^,]+),([^,]+),([^,]+),([^,]+),([^)]+)\)\)', line.rstrip('.'))
                if match:
                    x, y, h, k, a, b = [float(match.group(i)) for i in range(1, 7)]
                    dx, dy = x - h, y - k
                    neg_examples.append({
                        'x': x, 'y': y, 'h': h, 'k': k, 'a': a, 'b': b,
                        'dx': dx, 'dy': dy,
                        'dx_sq': dx*dx, 'dy_sq': dy*dy,
                        'ellipse_val': (dx*dx)/(a*a) + (dy*dy)/(b*b)
                    })
    
    X_list = pos_examples + neg_examples
    y_list = [1] * len(pos_examples) + [0] * len(neg_examples)
    
    X = pd.DataFrame(X_list)
    y = np.array(y_list)
    X.index = [f"e_{i+1}" for i in range(len(X))]
    
    return X, y, bk_file


def load_hyperbola_side_data(examples_file, bk_file=None):
    """Load hyperbola_side data: x² - y² > k"""
    pos_examples = []
    neg_examples = []
    
    with open(examples_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('hyperbola_side(') and not line.startswith('not('):
                match = re.match(r'hyperbola_side\(([^,]+),([^,]+),([^)]+)\)', line.rstrip('.'))
                if match:
                    x, y, k = float(match.group(1)), float(match.group(2)), float(match.group(3))
                    pos_examples.append({
                        'x': x, 'y': y, 'k': k,
                        'x_sq': x*x, 'y_sq': y*y,
                        'hyperbola_val': x*x - y*y
                    })
            elif line.startswith('not(hyperbola_side('):
                match = re.match(r'not\(hyperbola_side\(([^,]+),([^,]+),([^)]+)\)\)', line.rstrip('.'))
                if match:
                    x, y, k = float(match.group(1)), float(match.group(2)), float(match.group(3))
                    neg_examples.append({
                        'x': x, 'y': y, 'k': k,
                        'x_sq': x*x, 'y_sq': y*y,
                        'hyperbola_val': x*x - y*y
                    })
    
    X_list = pos_examples + neg_examples
    y_list = [1] * len(pos_examples) + [0] * len(neg_examples)
    
    X = pd.DataFrame(X_list)
    y = np.array(y_list)
    X.index = [f"e_{i+1}" for i in range(len(X))]
    
    return X, y, bk_file


def load_xy_less_than_data(examples_file, bk_file=None):
    """Load xy_less_than data: x·y < c"""
    pos_examples = []
    neg_examples = []
    
    with open(examples_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('xy_less_than(') and not line.startswith('not('):
                match = re.match(r'xy_less_than\(([^,]+),([^,]+),([^)]+)\)', line.rstrip('.'))
                if match:
                    x, y, c = float(match.group(1)), float(match.group(2)), float(match.group(3))
                    pos_examples.append({
                        'x': x, 'y': y, 'c': c,
                        'xy': x * y
                    })
            elif line.startswith('not(xy_less_than('):
                match = re.match(r'not\(xy_less_than\(([^,]+),([^,]+),([^)]+)\)\)', line.rstrip('.'))
                if match:
                    x, y, c = float(match.group(1)), float(match.group(2)), float(match.group(3))
                    neg_examples.append({
                        'x': x, 'y': y, 'c': c,
                        'xy': x * y
                    })
    
    X_list = pos_examples + neg_examples
    y_list = [1] * len(pos_examples) + [0] * len(neg_examples)
    
    X = pd.DataFrame(X_list)
    y = np.array(y_list)
    X.index = [f"e_{i+1}" for i in range(len(X))]
    
    return X, y, bk_file


def load_quad_strip_data(examples_file, bk_file=None):
    """Load quad_strip data: a < x² < b"""
    pos_examples = []
    neg_examples = []
    
    with open(examples_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('quad_strip(') and not line.startswith('not('):
                match = re.match(r'quad_strip\(([^,]+),([^,]+),([^,]+),([^)]+)\)', line.rstrip('.'))
                if match:
                    x, y, a, b = float(match.group(1)), float(match.group(2)), float(match.group(3)), float(match.group(4))
                    pos_examples.append({
                        'x': x, 'y': y, 'a': a, 'b': b,
                        'x_sq': x*x
                    })
            elif line.startswith('not(quad_strip('):
                match = re.match(r'not\(quad_strip\(([^,]+),([^,]+),([^,]+),([^)]+)\)\)', line.rstrip('.'))
                if match:
                    x, y, a, b = float(match.group(1)), float(match.group(2)), float(match.group(3)), float(match.group(4))
                    neg_examples.append({
                        'x': x, 'y': y, 'a': a, 'b': b,
                        'x_sq': x*x
                    })
    
    X_list = pos_examples + neg_examples
    y_list = [1] * len(pos_examples) + [0] * len(neg_examples)
    
    X = pd.DataFrame(X_list)
    y = np.array(y_list)
    X.index = [f"e_{i+1}" for i in range(len(X))]
    
    return X, y, bk_file


# ========== Category 2: Disjunctive (OR) Regions ==========

def load_union_halfplanes_data(examples_file, bk_file=None):
    """Load union_halfplanes data: (ax + by ≤ c) OR (dx + ey ≥ f)"""
    pos_examples = []
    neg_examples = []
    
    with open(examples_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('union_halfplanes(') and not line.startswith('not('):
                match = re.match(r'union_halfplanes\(([^,]+),([^)]+)\)', line.rstrip('.'))
                if match:
                    x, y = float(match.group(1)), float(match.group(2))
                    pos_examples.append({
                        'x': x, 'y': y,
                        'xy': x * y, 'x_sq': x*x, 'y_sq': y*y
                    })
            elif line.startswith('not(union_halfplanes('):
                match = re.match(r'not\(union_halfplanes\(([^,]+),([^)]+)\)\)', line.rstrip('.'))
                if match:
                    x, y = float(match.group(1)), float(match.group(2))
                    neg_examples.append({
                        'x': x, 'y': y,
                        'xy': x * y, 'x_sq': x*x, 'y_sq': y*y
                    })
    
    X_list = pos_examples + neg_examples
    y_list = [1] * len(pos_examples) + [0] * len(neg_examples)
    
    X = pd.DataFrame(X_list)
    y = np.array(y_list)
    X.index = [f"e_{i+1}" for i in range(len(X))]
    
    return X, y, bk_file


def load_circle_or_box_data(examples_file, bk_file=None):
    """Load circle_or_box data: (x² + y² < r²) OR (|x| < s AND |y| < s)"""
    pos_examples = []
    neg_examples = []
    
    with open(examples_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('circle_or_box(') and not line.startswith('not('):
                match = re.match(r'circle_or_box\(([^,]+),([^,]+),([^,]+),([^)]+)\)', line.rstrip('.'))
                if match:
                    x, y, r, box_size = float(match.group(1)), float(match.group(2)), float(match.group(3)), float(match.group(4))
                    pos_examples.append({
                        'x': x, 'y': y, 'r': r, 'box_size': box_size,
                        'x_sq': x*x, 'y_sq': y*y, 'dist_sq': x*x + y*y,
                        'abs_x': abs(x), 'abs_y': abs(y)
                    })
            elif line.startswith('not(circle_or_box('):
                match = re.match(r'not\(circle_or_box\(([^,]+),([^,]+),([^,]+),([^)]+)\)\)', line.rstrip('.'))
                if match:
                    x, y, r, box_size = float(match.group(1)), float(match.group(2)), float(match.group(3)), float(match.group(4))
                    neg_examples.append({
                        'x': x, 'y': y, 'r': r, 'box_size': box_size,
                        'x_sq': x*x, 'y_sq': y*y, 'dist_sq': x*x + y*y,
                        'abs_x': abs(x), 'abs_y': abs(y)
                    })
    
    X_list = pos_examples + neg_examples
    y_list = [1] * len(pos_examples) + [0] * len(neg_examples)
    
    X = pd.DataFrame(X_list)
    y = np.array(y_list)
    X.index = [f"e_{i+1}" for i in range(len(X))]
    
    return X, y, bk_file


def load_piecewise_data(examples_file, bk_file=None):
    """Load piecewise data: (x < 0 AND y > 5) OR (x > 10)"""
    pos_examples = []
    neg_examples = []
    
    with open(examples_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('piecewise(') and not line.startswith('not('):
                match = re.match(r'piecewise\(([^,]+),([^)]+)\)', line.rstrip('.'))
                if match:
                    x, y = float(match.group(1)), float(match.group(2))
                    pos_examples.append({
                        'x': x, 'y': y,
                        'x_sq': x*x, 'y_sq': y*y, 'xy': x*y
                    })
            elif line.startswith('not(piecewise('):
                match = re.match(r'not\(piecewise\(([^,]+),([^)]+)\)\)', line.rstrip('.'))
                if match:
                    x, y = float(match.group(1)), float(match.group(2))
                    neg_examples.append({
                        'x': x, 'y': y,
                        'x_sq': x*x, 'y_sq': y*y, 'xy': x*y
                    })
    
    X_list = pos_examples + neg_examples
    y_list = [1] * len(pos_examples) + [0] * len(neg_examples)
    
    X = pd.DataFrame(X_list)
    y = np.array(y_list)
    X.index = [f"e_{i+1}" for i in range(len(X))]
    
    return X, y, bk_file


def load_fallback_region_data(examples_file, bk_file=None):
    """Load fallback_region data: in_circle OR below_line"""
    pos_examples = []
    neg_examples = []
    
    with open(examples_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('fallback_region(') and not line.startswith('not('):
                match = re.match(r'fallback_region\(([^,]+),([^,]+),([^,]+),([^,]+),([^,]+),([^)]+)\)', line.rstrip('.'))
                if match:
                    x, y, r, a, b, c = [float(match.group(i)) for i in range(1, 7)]
                    pos_examples.append({
                        'x': x, 'y': y, 'r': r, 'a': a, 'b': b, 'c': c,
                        'x_sq': x*x, 'y_sq': y*y, 'dist_sq': x*x + y*y,
                        'line_val': a*x + b*y + c
                    })
            elif line.startswith('not(fallback_region('):
                match = re.match(r'not\(fallback_region\(([^,]+),([^,]+),([^,]+),([^,]+),([^,]+),([^)]+)\)\)', line.rstrip('.'))
                if match:
                    x, y, r, a, b, c = [float(match.group(i)) for i in range(1, 7)]
                    neg_examples.append({
                        'x': x, 'y': y, 'r': r, 'a': a, 'b': b, 'c': c,
                        'x_sq': x*x, 'y_sq': y*y, 'dist_sq': x*x + y*y,
                        'line_val': a*x + b*y + c
                    })
    
    X_list = pos_examples + neg_examples
    y_list = [1] * len(pos_examples) + [0] * len(neg_examples)
    
    X = pd.DataFrame(X_list)
    y = np.array(y_list)
    X.index = [f"e_{i+1}" for i in range(len(X))]
    
    return X, y, bk_file


# ========== Category 3: Non-convex / Piecewise / Hybrid Regions ==========

def load_donut_data(examples_file, bk_file=None):
    """Load donut data: r₁² < x² + y² < r₂²"""
    pos_examples = []
    neg_examples = []
    
    with open(examples_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('donut(') and not line.startswith('not('):
                match = re.match(r'donut\(([^,]+),([^,]+),([^,]+),([^)]+)\)', line.rstrip('.'))
                if match:
                    x, y, r1, r2 = float(match.group(1)), float(match.group(2)), float(match.group(3)), float(match.group(4))
                    pos_examples.append({
                        'x': x, 'y': y, 'r1': r1, 'r2': r2,
                        'x_sq': x*x, 'y_sq': y*y, 'dist_sq': x*x + y*y
                    })
            elif line.startswith('not(donut('):
                match = re.match(r'not\(donut\(([^,]+),([^,]+),([^,]+),([^)]+)\)\)', line.rstrip('.'))
                if match:
                    x, y, r1, r2 = float(match.group(1)), float(match.group(2)), float(match.group(3)), float(match.group(4))
                    neg_examples.append({
                        'x': x, 'y': y, 'r1': r1, 'r2': r2,
                        'x_sq': x*x, 'y_sq': y*y, 'dist_sq': x*x + y*y
                    })
    
    X_list = pos_examples + neg_examples
    y_list = [1] * len(pos_examples) + [0] * len(neg_examples)
    
    X = pd.DataFrame(X_list)
    y = np.array(y_list)
    X.index = [f"e_{i+1}" for i in range(len(X))]
    
    return X, y, bk_file


def load_lshape_data(examples_file, bk_file=None):
    """Load lshape data: Union of two rectangles"""
    pos_examples = []
    neg_examples = []
    
    with open(examples_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('lshape(') and not line.startswith('not('):
                match = re.match(r'lshape\(([^,]+),([^)]+)\)', line.rstrip('.'))
                if match:
                    x, y = float(match.group(1)), float(match.group(2))
                    pos_examples.append({
                        'x': x, 'y': y,
                        'x_sq': x*x, 'y_sq': y*y, 'xy': x*y
                    })
            elif line.startswith('not(lshape('):
                match = re.match(r'not\(lshape\(([^,]+),([^)]+)\)\)', line.rstrip('.'))
                if match:
                    x, y = float(match.group(1)), float(match.group(2))
                    neg_examples.append({
                        'x': x, 'y': y,
                        'x_sq': x*x, 'y_sq': y*y, 'xy': x*y
                    })
    
    X_list = pos_examples + neg_examples
    y_list = [1] * len(pos_examples) + [0] * len(neg_examples)
    
    X = pd.DataFrame(X_list)
    y = np.array(y_list)
    X.index = [f"e_{i+1}" for i in range(len(X))]
    
    return X, y, bk_file


def load_above_parabola_data(examples_file, bk_file=None):
    """Load above_parabola data: y > ax² + bx + c"""
    pos_examples = []
    neg_examples = []
    
    with open(examples_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('above_parabola(') and not line.startswith('not('):
                match = re.match(r'above_parabola\(([^,]+),([^,]+),([^,]+),([^,]+),([^)]+)\)', line.rstrip('.'))
                if match:
                    x, y, a, b, c = float(match.group(1)), float(match.group(2)), float(match.group(3)), float(match.group(4)), float(match.group(5))
                    pos_examples.append({
                        'x': x, 'y': y, 'a': a, 'b': b, 'c': c,
                        'x_sq': x*x, 'y_sq': y*y,
                        'parabola_val': a*x*x + b*x + c
                    })
            elif line.startswith('not(above_parabola('):
                match = re.match(r'not\(above_parabola\(([^,]+),([^,]+),([^,]+),([^,]+),([^)]+)\)\)', line.rstrip('.'))
                if match:
                    x, y, a, b, c = float(match.group(1)), float(match.group(2)), float(match.group(3)), float(match.group(4)), float(match.group(5))
                    neg_examples.append({
                        'x': x, 'y': y, 'a': a, 'b': b, 'c': c,
                        'x_sq': x*x, 'y_sq': y*y,
                        'parabola_val': a*x*x + b*x + c
                    })
    
    X_list = pos_examples + neg_examples
    y_list = [1] * len(pos_examples) + [0] * len(neg_examples)
    
    X = pd.DataFrame(X_list)
    y = np.array(y_list)
    X.index = [f"e_{i+1}" for i in range(len(X))]
    
    return X, y, bk_file


def load_sinusoidal_data(examples_file, bk_file=None):
    """Load sinusoidal data: y > sin(scale·x) + offset"""
    pos_examples = []
    neg_examples = []
    
    with open(examples_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('sinusoidal(') and not line.startswith('not('):
                match = re.match(r'sinusoidal\(([^,]+),([^,]+),([^,]+),([^)]+)\)', line.rstrip('.'))
                if match:
                    x, y, scale, offset = float(match.group(1)), float(match.group(2)), float(match.group(3)), float(match.group(4))
                    pos_examples.append({
                        'x': x, 'y': y, 'scale': scale, 'offset': offset,
                        'x_sq': x*x, 'y_sq': y*y,
                        'sin_val': math.sin(scale * x) + offset
                    })
            elif line.startswith('not(sinusoidal('):
                match = re.match(r'not\(sinusoidal\(([^,]+),([^,]+),([^,]+),([^)]+)\)\)', line.rstrip('.'))
                if match:
                    x, y, scale, offset = float(match.group(1)), float(match.group(2)), float(match.group(3)), float(match.group(4))
                    neg_examples.append({
                        'x': x, 'y': y, 'scale': scale, 'offset': offset,
                        'x_sq': x*x, 'y_sq': y*y,
                        'sin_val': math.sin(scale * x) + offset
                    })
    
    X_list = pos_examples + neg_examples
    y_list = [1] * len(pos_examples) + [0] * len(neg_examples)
    
    X = pd.DataFrame(X_list)
    y = np.array(y_list)
    X.index = [f"e_{i+1}" for i in range(len(X))]
    
    return X, y, bk_file


def load_crescent_data(examples_file, bk_file=None):
    """Load crescent data: (x² + y² < r₁²) AND ((x-h)² + (y-k)² > r₂²)"""
    pos_examples = []
    neg_examples = []
    
    with open(examples_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('crescent(') and not line.startswith('not('):
                match = re.match(r'crescent\(([^,]+),([^,]+),([^,]+),([^,]+),([^,]+),([^)]+)\)', line.rstrip('.'))
                if match:
                    x, y, r1, h, k, r2 = [float(match.group(i)) for i in range(1, 7)]
                    dx, dy = x - h, y - k
                    pos_examples.append({
                        'x': x, 'y': y, 'r1': r1, 'h': h, 'k': k, 'r2': r2,
                        'x_sq': x*x, 'y_sq': y*y, 'dist1_sq': x*x + y*y,
                        'dx': dx, 'dy': dy, 'dist2_sq': dx*dx + dy*dy
                    })
            elif line.startswith('not(crescent('):
                match = re.match(r'not\(crescent\(([^,]+),([^,]+),([^,]+),([^,]+),([^,]+),([^)]+)\)\)', line.rstrip('.'))
                if match:
                    x, y, r1, h, k, r2 = [float(match.group(i)) for i in range(1, 7)]
                    dx, dy = x - h, y - k
                    neg_examples.append({
                        'x': x, 'y': y, 'r1': r1, 'h': h, 'k': k, 'r2': r2,
                        'x_sq': x*x, 'y_sq': y*y, 'dist1_sq': x*x + y*y,
                        'dx': dx, 'dy': dy, 'dist2_sq': dx*dx + dy*dy
                    })
    
    X_list = pos_examples + neg_examples
    y_list = [1] * len(pos_examples) + [0] * len(neg_examples)
    
    X = pd.DataFrame(X_list)
    y = np.array(y_list)
    X.index = [f"e_{i+1}" for i in range(len(X))]
    
    return X, y, bk_file
