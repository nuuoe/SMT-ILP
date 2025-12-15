#!/usr/bin/env python3
"""
Load geometry2 problems from Prolog files
Geometry2: Relational Spatial Constraints
"""

import re
import pandas as pd
import numpy as np
import os


def parse_point(point_str):
    """Parse point string 'x,y' or 'x,y,z' into list of floats"""
    return [float(x.strip()) for x in point_str.split(',')]


def load_left_of_data(examples_file, bk_file=None):
    """Load left_of data: left_of(A, B) where A and B are 2D points"""
    pos_examples = []
    neg_examples = []
    
    with open(examples_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('left_of(') and not line.startswith('not('):
                # Positive example: left_of(x1,y1,x2,y2)
                match = re.match(r'left_of\(([^,]+),([^,]+),([^,]+),([^)]+)\)', line.rstrip('.'))
                if match:
                    a_x, a_y = float(match.group(1)), float(match.group(2))
                    b_x, b_y = float(match.group(3)), float(match.group(4))
                    pos_examples.append({
                        'a_x': a_x, 'a_y': a_y,
                        'b_x': b_x, 'b_y': b_y,
                        'diff_x': b_x - a_x  # Feature: difference in x coordinates
                    })
            elif line.startswith('not(left_of('):
                # Negative example
                match = re.match(r'not\(left_of\(([^,]+),([^,]+),([^,]+),([^)]+)\)\)', line.rstrip('.'))
                if match:
                    a_x, a_y = float(match.group(1)), float(match.group(2))
                    b_x, b_y = float(match.group(3)), float(match.group(4))
                    neg_examples.append({
                        'a_x': a_x, 'a_y': a_y,
                        'b_x': b_x, 'b_y': b_y,
                        'diff_x': b_x - a_x
                    })
    
    X_list = pos_examples + neg_examples
    y_list = [1] * len(pos_examples) + [0] * len(neg_examples)
    
    X = pd.DataFrame(X_list)
    y = np.array(y_list)
    X.index = [f"e_{i+1}" for i in range(len(X))]
    
    return X, y, bk_file


def load_closer_than_data(examples_file, bk_file=None):
    """Load closer_than data: closer_than(A, B, C) where A, B, C are 2D points"""
    pos_examples = []
    neg_examples = []
    
    def distance_2d(p1, p2):
        return ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)**0.5
    
    with open(examples_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('closer_than(') and not line.startswith('not('):
                # Positive example: closer_than(ax,ay,bx,by,cx,cy)
                match = re.match(r'closer_than\(([^,]+),([^,]+),([^,]+),([^,]+),([^,]+),([^)]+)\)', line.rstrip('.'))
                if match:
                    a = [float(match.group(1)), float(match.group(2))]
                    b = [float(match.group(3)), float(match.group(4))]
                    c = [float(match.group(5)), float(match.group(6))]
                    dist_ab = distance_2d(a, b)
                    dist_cb = distance_2d(c, b)
                    dist_diff = dist_cb - dist_ab
                    
                    # Additional features for better learning
                    dist_ratio = dist_ab / (dist_cb + 1e-6)  # Avoid division by zero
                    dist_sq_diff = dist_cb**2 - dist_ab**2  # Squared distance difference
                    
                    pos_examples.append({
                        'a_x': a[0], 'a_y': a[1],
                        'b_x': b[0], 'b_y': b[1],
                        'c_x': c[0], 'c_y': c[1],
                        'dist_ab': dist_ab,
                        'dist_cb': dist_cb,
                        'dist_diff': dist_diff,
                        'dist_ratio': dist_ratio,
                        'dist_sq_diff': dist_sq_diff
                    })
            elif line.startswith('not(closer_than('):
                # Negative example
                match = re.match(r'not\(closer_than\(([^,]+),([^,]+),([^,]+),([^,]+),([^,]+),([^)]+)\)\)', line.rstrip('.'))
                if match:
                    a = [float(match.group(1)), float(match.group(2))]
                    b = [float(match.group(3)), float(match.group(4))]
                    c = [float(match.group(5)), float(match.group(6))]
                    dist_ab = distance_2d(a, b)
                    dist_cb = distance_2d(c, b)
                    dist_diff = dist_cb - dist_ab
                    dist_ratio = dist_ab / (dist_cb + 1e-6)
                    dist_sq_diff = dist_cb**2 - dist_ab**2
                    
                    neg_examples.append({
                        'a_x': a[0], 'a_y': a[1],
                        'b_x': b[0], 'b_y': b[1],
                        'c_x': c[0], 'c_y': c[1],
                        'dist_ab': dist_ab,
                        'dist_cb': dist_cb,
                        'dist_diff': dist_diff,
                        'dist_ratio': dist_ratio,
                        'dist_sq_diff': dist_sq_diff
                    })
    
    X_list = pos_examples + neg_examples
    y_list = [1] * len(pos_examples) + [0] * len(neg_examples)
    
    X = pd.DataFrame(X_list)
    y = np.array(y_list)
    X.index = [f"e_{i+1}" for i in range(len(X))]
    
    return X, y, bk_file


def load_touching_data(examples_file, bk_file=None):
    """Load touching data: touching(A, B) where A and B are 2D points"""
    pos_examples = []
    neg_examples = []
    
    def distance_2d(p1, p2):
        return ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)**0.5
    
    with open(examples_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('touching(') and not line.startswith('not('):
                # Positive example: touching(ax,ay,bx,by)
                match = re.match(r'touching\(([^,]+),([^,]+),([^,]+),([^)]+)\)', line.rstrip('.'))
                if match:
                    a = [float(match.group(1)), float(match.group(2))]
                    b = [float(match.group(3)), float(match.group(4))]
                    dist = distance_2d(a, b)
                    pos_examples.append({
                        'a_x': a[0], 'a_y': a[1],
                        'b_x': b[0], 'b_y': b[1],
                        'distance': dist
                    })
            elif line.startswith('not(touching('):
                # Negative example
                match = re.match(r'not\(touching\(([^,]+),([^,]+),([^,]+),([^)]+)\)\)', line.rstrip('.'))
                if match:
                    a = [float(match.group(1)), float(match.group(2))]
                    b = [float(match.group(3)), float(match.group(4))]
                    dist = distance_2d(a, b)
                    neg_examples.append({
                        'a_x': a[0], 'a_y': a[1],
                        'b_x': b[0], 'b_y': b[1],
                        'distance': dist
                    })
    
    X_list = pos_examples + neg_examples
    y_list = [1] * len(pos_examples) + [0] * len(neg_examples)
    
    X = pd.DataFrame(X_list)
    y = np.array(y_list)
    X.index = [f"e_{i+1}" for i in range(len(X))]
    
    return X, y, bk_file


def load_inside_data(examples_file, bk_file=None):
    """Load inside data: inside(A, B) where A is a point and B is a bounding box"""
    pos_examples = []
    neg_examples = []
    
    with open(examples_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('inside(') and not line.startswith('not('):
                # Positive example: inside(ax,ay,bx_min,by_min,bx_max,by_max)
                match = re.match(r'inside\(([^,]+),([^,]+),([^,]+),([^,]+),([^,]+),([^)]+)\)', line.rstrip('.'))
                if match:
                    a_x = float(match.group(1))
                    a_y = float(match.group(2))
                    b_x_min = float(match.group(3))
                    b_y_min = float(match.group(4))
                    b_x_max = float(match.group(5))
                    b_y_max = float(match.group(6))
                    # Compute margins from each edge
                    left_margin = a_x - b_x_min
                    right_margin = b_x_max - a_x
                    bottom_margin = a_y - b_y_min
                    top_margin = b_y_max - a_y
                    
                    # Distance to nearest edge
                    min_edge_dist = min(left_margin, right_margin, bottom_margin, top_margin)
                    
                    # Relative position within box (normalized 0-1)
                    box_width = b_x_max - b_x_min
                    box_height = b_y_max - b_y_min
                    rel_x = (a_x - b_x_min) / (box_width + 1e-6) if box_width > 1e-6 else 0.5
                    rel_y = (a_y - b_y_min) / (box_height + 1e-6) if box_height > 1e-6 else 0.5
                    
                    pos_examples.append({
                        'a_x': a_x, 'a_y': a_y,
                        'b_x_min': b_x_min, 'b_y_min': b_y_min,
                        'b_x_max': b_x_max, 'b_y_max': b_y_max,
                        'a_x_in': 1 if (b_x_min < a_x < b_x_max) else 0,
                        'a_y_in': 1 if (b_y_min < a_y < b_y_max) else 0,
                        'left_margin': left_margin,
                        'right_margin': right_margin,
                        'bottom_margin': bottom_margin,
                        'top_margin': top_margin,
                        'min_edge_dist': min_edge_dist,
                        'rel_x': rel_x,
                        'rel_y': rel_y
                    })
            elif line.startswith('not(inside('):
                # Negative example
                match = re.match(r'not\(inside\(([^,]+),([^,]+),([^,]+),([^,]+),([^,]+),([^)]+)\)\)', line.rstrip('.'))
                if match:
                    a_x = float(match.group(1))
                    a_y = float(match.group(2))
                    b_x_min = float(match.group(3))
                    b_y_min = float(match.group(4))
                    b_x_max = float(match.group(5))
                    b_y_max = float(match.group(6))
                    left_margin = a_x - b_x_min
                    right_margin = b_x_max - a_x
                    bottom_margin = a_y - b_y_min
                    top_margin = b_y_max - a_y
                    
                    min_edge_dist = min(left_margin, right_margin, bottom_margin, top_margin)
                    
                    box_width = b_x_max - b_x_min
                    box_height = b_y_max - b_y_min
                    rel_x = (a_x - b_x_min) / (box_width + 1e-6) if box_width > 1e-6 else 0.5
                    rel_y = (a_y - b_y_min) / (box_height + 1e-6) if box_height > 1e-6 else 0.5
                    
                    neg_examples.append({
                        'a_x': a_x, 'a_y': a_y,
                        'b_x_min': b_x_min, 'b_y_min': b_y_min,
                        'b_x_max': b_x_max, 'b_y_max': b_y_max,
                        'a_x_in': 1 if (b_x_min < a_x < b_x_max) else 0,
                        'a_y_in': 1 if (b_y_min < a_y < b_y_max) else 0,
                        'left_margin': left_margin,
                        'right_margin': right_margin,
                        'bottom_margin': bottom_margin,
                        'top_margin': top_margin,
                        'min_edge_dist': min_edge_dist,
                        'rel_x': rel_x,
                        'rel_y': rel_y
                    })
    
    X_list = pos_examples + neg_examples
    y_list = [1] * len(pos_examples) + [0] * len(neg_examples)
    
    X = pd.DataFrame(X_list)
    y = np.array(y_list)
    X.index = [f"e_{i+1}" for i in range(len(X))]
    
    return X, y, bk_file
def load_overlapping_data(examples_file, bk_file=None):
    """Load overlapping data: overlapping(A, B) where A and B are bounding boxes"""
    pos_examples = []
    neg_examples = []
    
    def boxes_overlap(box1, box2):
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2
        x_overlap = not (x1_max < x2_min or x2_max < x1_min)
        y_overlap = not (y1_max < y2_min or y2_max < y1_min)
        return x_overlap and y_overlap
    
    with open(examples_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('overlapping(') and not line.startswith('not('):
                match = re.match(r'overlapping\(([^,]+),([^,]+),([^,]+),([^,]+),([^,]+),([^,]+),([^,]+),([^)]+)\)', line.rstrip('.'))
                if match:
                    a_x_min, a_y_min = float(match.group(1)), float(match.group(2))
                    a_x_max, a_y_max = float(match.group(3)), float(match.group(4))
                    b_x_min, b_y_min = float(match.group(5)), float(match.group(6))
                    b_x_max, b_y_max = float(match.group(7)), float(match.group(8))
                    box_a = [a_x_min, a_y_min, a_x_max, a_y_max]
                    box_b = [b_x_min, b_y_min, b_x_max, b_y_max]
                    pos_examples.append({
                        'a_x_min': a_x_min, 'a_y_min': a_y_min, 'a_x_max': a_x_max, 'a_y_max': a_y_max,
                        'b_x_min': b_x_min, 'b_y_min': b_y_min, 'b_x_max': b_x_max, 'b_y_max': b_y_max,
                        'overlaps': 1 if boxes_overlap(box_a, box_b) else 0
                    })
            elif line.startswith('not(overlapping('):
                match = re.match(r'not\(overlapping\(([^,]+),([^,]+),([^,]+),([^,]+),([^,]+),([^,]+),([^,]+),([^)]+)\)\)', line.rstrip('.'))
                if match:
                    a_x_min, a_y_min = float(match.group(1)), float(match.group(2))
                    a_x_max, a_y_max = float(match.group(3)), float(match.group(4))
                    b_x_min, b_y_min = float(match.group(5)), float(match.group(6))
                    b_x_max, b_y_max = float(match.group(7)), float(match.group(8))
                    box_a = [a_x_min, a_y_min, a_x_max, a_y_max]
                    box_b = [b_x_min, b_y_min, b_x_max, b_y_max]
                    neg_examples.append({
                        'a_x_min': a_x_min, 'a_y_min': a_y_min, 'a_x_max': a_x_max, 'a_y_max': a_y_max,
                        'b_x_min': b_x_min, 'b_y_min': b_y_min, 'b_x_max': b_x_max, 'b_y_max': b_y_max,
                        'overlaps': 1 if boxes_overlap(box_a, box_b) else 0
                    })
    
    X_list = pos_examples + neg_examples
    y_list = [1] * len(pos_examples) + [0] * len(neg_examples)
    
    X = pd.DataFrame(X_list)
    y = np.array(y_list)
    X.index = [f"e_{i+1}" for i in range(len(X))]
    
    return X, y, bk_file


def load_between_data(examples_file, bk_file=None):
    """Load between data: between(A, B, C) where A, B, C are points"""
    pos_examples = []
    neg_examples = []
    
    def distance_2d(p1, p2):
        return ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)**0.5
    
    def cross_product(p1, p2, p3):
        """Compute cross product for collinearity check"""
        return (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0])
    
    with open(examples_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('between(') and not line.startswith('not('):
                match = re.match(r'between\(([^,]+),([^,]+),([^,]+),([^,]+),([^,]+),([^)]+)\)', line.rstrip('.'))
                if match:
                    a_x, a_y = float(match.group(1)), float(match.group(2))
                    b_x, b_y = float(match.group(3)), float(match.group(4))
                    c_x, c_y = float(match.group(5)), float(match.group(6))
                    a = [a_x, a_y]
                    b = [b_x, b_y]
                    c = [c_x, c_y]
                    dist_ab = distance_2d(a, b)
                    dist_ac = distance_2d(a, c)
                    dist_bc = distance_2d(b, c)
                    
                    # Cross product for collinearity (should be near 0)
                    cross = abs(cross_product(b, a, c))
                    
                    # Check if A is on the line segment from B to C
                    # Parameter t: A = B + t*(C - B), where 0 <= t <= 1
                    # Compute t for x and y separately
                    if abs(c_x - b_x) > 1e-6:
                        t_x = (a_x - b_x) / (c_x - b_x)
                    else:
                        t_x = (a_y - b_y) / (c_y - b_y) if abs(c_y - b_y) > 1e-6 else 0.5
                    
                    if abs(c_y - b_y) > 1e-6:
                        t_y = (a_y - b_y) / (c_y - b_y)
                    else:
                        t_y = (a_x - b_x) / (c_x - b_x) if abs(c_x - b_x) > 1e-6 else 0.5
                    
                    t_avg = (t_x + t_y) / 2.0  # Average parameter
                    
                    pos_examples.append({
                        'a_x': a_x, 'a_y': a_y,
                        'b_x': b_x, 'b_y': b_y,
                        'c_x': c_x, 'c_y': c_y,
                        'dist_ab': dist_ab, 'dist_ac': dist_ac, 'dist_bc': dist_bc,
                        'cross_product': cross,
                        't_parameter': t_avg
                    })
            elif line.startswith('not(between('):
                match = re.match(r'not\(between\(([^,]+),([^,]+),([^,]+),([^,]+),([^,]+),([^)]+)\)\)', line.rstrip('.'))
                if match:
                    a_x, a_y = float(match.group(1)), float(match.group(2))
                    b_x, b_y = float(match.group(3)), float(match.group(4))
                    c_x, c_y = float(match.group(5)), float(match.group(6))
                    a = [a_x, a_y]
                    b = [b_x, b_y]
                    c = [c_x, c_y]
                    dist_ab = distance_2d(a, b)
                    dist_ac = distance_2d(a, c)
                    dist_bc = distance_2d(b, c)
                    
                    cross = abs(cross_product(b, a, c))
                    
                    if abs(c_x - b_x) > 1e-6:
                        t_x = (a_x - b_x) / (c_x - b_x)
                    else:
                        t_x = (a_y - b_y) / (c_y - b_y) if abs(c_y - b_y) > 1e-6 else 0.5
                    
                    if abs(c_y - b_y) > 1e-6:
                        t_y = (a_y - b_y) / (c_y - b_y)
                    else:
                        t_y = (a_x - b_x) / (c_x - b_x) if abs(c_x - b_x) > 1e-6 else 0.5
                    
                    t_avg = (t_x + t_y) / 2.0
                    
                    neg_examples.append({
                        'a_x': a_x, 'a_y': a_y,
                        'b_x': b_x, 'b_y': b_y,
                        'c_x': c_x, 'c_y': c_y,
                        'dist_ab': dist_ab, 'dist_ac': dist_ac, 'dist_bc': dist_bc,
                        'cross_product': cross,
                        't_parameter': t_avg
                    })
    
    X_list = pos_examples + neg_examples
    y_list = [1] * len(pos_examples) + [0] * len(neg_examples)
    
    X = pd.DataFrame(X_list)
    y = np.array(y_list)
    X.index = [f"e_{i+1}" for i in range(len(X))]
    
    return X, y, bk_file


def load_adjacent_data(examples_file, bk_file=None):
    """Load adjacent data: adjacent(A, B) where A and B are bounding boxes"""
    pos_examples = []
    neg_examples = []
    
    with open(examples_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('adjacent(') and not line.startswith('not('):
                match = re.match(r'adjacent\(([^,]+),([^,]+),([^,]+),([^,]+),([^,]+),([^,]+),([^,]+),([^)]+)\)', line.rstrip('.'))
                if match:
                    a_x_min, a_y_min = float(match.group(1)), float(match.group(2))
                    a_x_max, a_y_max = float(match.group(3)), float(match.group(4))
                    b_x_min, b_y_min = float(match.group(5)), float(match.group(6))
                    b_x_max, b_y_max = float(match.group(7)), float(match.group(8))
                    
                    # Compute edge distances for adjacent detection
                    # Horizontal gap (for horizontal adjacency)
                    h_gap1 = abs(a_x_max - b_x_min)  # A right to B left
                    h_gap2 = abs(b_x_max - a_x_min)  # B right to A left
                    h_gap = min(h_gap1, h_gap2)
                    
                    # Vertical gap (for vertical adjacency)
                    v_gap1 = abs(a_y_max - b_y_min)  # A top to B bottom
                    v_gap2 = abs(b_y_max - a_y_min)  # B top to A bottom
                    v_gap = min(v_gap1, v_gap2)
                    
                    # Overlap indicators
                    x_overlap = not (a_x_max < b_x_min or b_x_max < a_x_min)
                    y_overlap = not (a_y_max < b_y_min or b_y_max < a_y_min)
                    
                    pos_examples.append({
                        'a_x_min': a_x_min, 'a_y_min': a_y_min, 'a_x_max': a_x_max, 'a_y_max': a_y_max,
                        'b_x_min': b_x_min, 'b_y_min': b_y_min, 'b_x_max': b_x_max, 'b_y_max': b_y_max,
                        'h_gap': h_gap, 'v_gap': v_gap,
                        'x_overlap': 1 if x_overlap else 0,
                        'y_overlap': 1 if y_overlap else 0
                    })
            elif line.startswith('not(adjacent('):
                match = re.match(r'not\(adjacent\(([^,]+),([^,]+),([^,]+),([^,]+),([^,]+),([^,]+),([^,]+),([^)]+)\)\)', line.rstrip('.'))
                if match:
                    a_x_min, a_y_min = float(match.group(1)), float(match.group(2))
                    a_x_max, a_y_max = float(match.group(3)), float(match.group(4))
                    b_x_min, b_y_min = float(match.group(5)), float(match.group(6))
                    b_x_max, b_y_max = float(match.group(7)), float(match.group(8))
                    
                    h_gap1 = abs(a_x_max - b_x_min)
                    h_gap2 = abs(b_x_max - a_x_min)
                    h_gap = min(h_gap1, h_gap2)
                    
                    v_gap1 = abs(a_y_max - b_y_min)
                    v_gap2 = abs(b_y_max - a_y_min)
                    v_gap = min(v_gap1, v_gap2)
                    
                    x_overlap = not (a_x_max < b_x_min or b_x_max < a_x_min)
                    y_overlap = not (a_y_max < b_y_min or b_y_max < a_y_min)
                    
                    neg_examples.append({
                        'a_x_min': a_x_min, 'a_y_min': a_y_min, 'a_x_max': a_x_max, 'a_y_max': a_y_max,
                        'b_x_min': b_x_min, 'b_y_min': b_y_min, 'b_x_max': b_x_max, 'b_y_max': b_y_max,
                        'h_gap': h_gap, 'v_gap': v_gap,
                        'x_overlap': 1 if x_overlap else 0,
                        'y_overlap': 1 if y_overlap else 0
                    })
    
    X_list = pos_examples + neg_examples
    y_list = [1] * len(pos_examples) + [0] * len(neg_examples)
    
    X = pd.DataFrame(X_list)
    y = np.array(y_list)
    X.index = [f"e_{i+1}" for i in range(len(X))]
    
    return X, y, bk_file


def load_aligned_data(examples_file, bk_file=None):
    """Load aligned data: aligned(A, B, C) where A, B, C are points"""
    pos_examples = []
    neg_examples = []
    
    def distance_2d(p1, p2):
        return ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)**0.5
    
    def cross_product(p1, p2, p3):
        """Compute cross product for collinearity check"""
        return (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0])
    
    with open(examples_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('aligned(') and not line.startswith('not('):
                match = re.match(r'aligned\(([^,]+),([^,]+),([^,]+),([^,]+),([^,]+),([^)]+)\)', line.rstrip('.'))
                if match:
                    a_x, a_y = float(match.group(1)), float(match.group(2))
                    b_x, b_y = float(match.group(3)), float(match.group(4))
                    c_x, c_y = float(match.group(5)), float(match.group(6))
                    a = [a_x, a_y]
                    b = [b_x, b_y]
                    c = [c_x, c_y]
                    dist_ab = distance_2d(a, b)
                    dist_ac = distance_2d(a, c)
                    dist_bc = distance_2d(b, c)
                    
                    # Cross product for collinearity (should be near 0)
                    cross = abs(cross_product(a, b, c))
                    
                    # Slope-based features
                    if abs(b_x - a_x) > 0.01:  # Avoid division by zero
                        slope_ab = (b_y - a_y) / (b_x - a_x)
                    else:
                        slope_ab = float('inf') if (b_y - a_y) != 0 else 0.0
                    
                    if abs(c_x - a_x) > 0.01:
                        slope_ac = (c_y - a_y) / (c_x - a_x)
                    else:
                        slope_ac = float('inf') if (c_y - a_y) != 0 else 0.0
                    
                    # Area of triangle (should be near 0 for collinear)
                    area = abs(cross) / 2.0
                    
                    pos_examples.append({
                        'a_x': a_x, 'a_y': a_y,
                        'b_x': b_x, 'b_y': b_y,
                        'c_x': c_x, 'c_y': c_y,
                        'dist_ab': dist_ab, 'dist_ac': dist_ac, 'dist_bc': dist_bc,
                        'cross_product': cross,
                        'area': area,
                        'slope_ab': slope_ab if slope_ab != float('inf') else 999.0,
                        'slope_ac': slope_ac if slope_ac != float('inf') else 999.0
                    })
            elif line.startswith('not(aligned('):
                match = re.match(r'not\(aligned\(([^,]+),([^,]+),([^,]+),([^,]+),([^,]+),([^)]+)\)\)', line.rstrip('.'))
                if match:
                    a_x, a_y = float(match.group(1)), float(match.group(2))
                    b_x, b_y = float(match.group(3)), float(match.group(4))
                    c_x, c_y = float(match.group(5)), float(match.group(6))
                    a = [a_x, a_y]
                    b = [b_x, b_y]
                    c = [c_x, c_y]
                    dist_ab = distance_2d(a, b)
                    dist_ac = distance_2d(a, c)
                    dist_bc = distance_2d(b, c)
                    
                    cross = abs(cross_product(a, b, c))
                    
                    if abs(b_x - a_x) > 0.01:
                        slope_ab = (b_y - a_y) / (b_x - a_x)
                    else:
                        slope_ab = float('inf') if (b_y - a_y) != 0 else 0.0
                    
                    if abs(c_x - a_x) > 0.01:
                        slope_ac = (c_y - a_y) / (c_x - a_x)
                    else:
                        slope_ac = float('inf') if (c_y - a_y) != 0 else 0.0
                    
                    area = abs(cross) / 2.0
                    
                    neg_examples.append({
                        'a_x': a_x, 'a_y': a_y,
                        'b_x': b_x, 'b_y': b_y,
                        'c_x': c_x, 'c_y': c_y,
                        'dist_ab': dist_ab, 'dist_ac': dist_ac, 'dist_bc': dist_bc,
                        'cross_product': cross,
                        'area': area,
                        'slope_ab': slope_ab if slope_ab != float('inf') else 999.0,
                        'slope_ac': slope_ac if slope_ac != float('inf') else 999.0
                    })
    
    X_list = pos_examples + neg_examples
    y_list = [1] * len(pos_examples) + [0] * len(neg_examples)
    
    X = pd.DataFrame(X_list)
    y = np.array(y_list)
    X.index = [f"e_{i+1}" for i in range(len(X))]
    
    return X, y, bk_file


def load_surrounds_data(examples_file, bk_file=None):
    """Load surrounds data: surrounds(A, B) where A and B are bounding boxes"""
    pos_examples = []
    neg_examples = []
    
    def box_surrounds(outer, inner):
        """Check if outer box surrounds inner box"""
        ox_min, oy_min, ox_max, oy_max = outer
        ix_min, iy_min, ix_max, iy_max = inner
        return (ox_min < ix_min and ix_max < ox_max and
                oy_min < iy_min and iy_max < oy_max)
    
    with open(examples_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('surrounds(') and not line.startswith('not('):
                match = re.match(r'surrounds\(([^,]+),([^,]+),([^,]+),([^,]+),([^,]+),([^,]+),([^,]+),([^)]+)\)', line.rstrip('.'))
                if match:
                    a_x_min, a_y_min = float(match.group(1)), float(match.group(2))
                    a_x_max, a_y_max = float(match.group(3)), float(match.group(4))
                    b_x_min, b_y_min = float(match.group(5)), float(match.group(6))
                    b_x_max, b_y_max = float(match.group(7)), float(match.group(8))
                    
                    box_a = [a_x_min, a_y_min, a_x_max, a_y_max]
                    box_b = [b_x_min, b_y_min, b_x_max, b_y_max]
                    
                    # Containment margins (how much padding)
                    left_margin = b_x_min - a_x_min
                    right_margin = a_x_max - b_x_max
                    bottom_margin = b_y_min - a_y_min
                    top_margin = a_y_max - b_y_max
                    
                    # Check if fully contained
                    fully_contained = box_surrounds(box_a, box_b)
                    
                    pos_examples.append({
                        'a_x_min': a_x_min, 'a_y_min': a_y_min, 'a_x_max': a_x_max, 'a_y_max': a_y_max,
                        'b_x_min': b_x_min, 'b_y_min': b_y_min, 'b_x_max': b_x_max, 'b_y_max': b_y_max,
                        'left_margin': left_margin,
                        'right_margin': right_margin,
                        'bottom_margin': bottom_margin,
                        'top_margin': top_margin,
                        'fully_contained': 1 if fully_contained else 0
                    })
            elif line.startswith('not(surrounds('):
                match = re.match(r'not\(surrounds\(([^,]+),([^,]+),([^,]+),([^,]+),([^,]+),([^,]+),([^,]+),([^)]+)\)\)', line.rstrip('.'))
                if match:
                    a_x_min, a_y_min = float(match.group(1)), float(match.group(2))
                    a_x_max, a_y_max = float(match.group(3)), float(match.group(4))
                    b_x_min, b_y_min = float(match.group(5)), float(match.group(6))
                    b_x_max, b_y_max = float(match.group(7)), float(match.group(8))
                    
                    box_a = [a_x_min, a_y_min, a_x_max, a_y_max]
                    box_b = [b_x_min, b_y_min, b_x_max, b_y_max]
                    
                    left_margin = b_x_min - a_x_min
                    right_margin = a_x_max - b_x_max
                    bottom_margin = b_y_min - a_y_min
                    top_margin = a_y_max - b_y_max
                    
                    fully_contained = box_surrounds(box_a, box_b)
                    
                    neg_examples.append({
                        'a_x_min': a_x_min, 'a_y_min': a_y_min, 'a_x_max': a_x_max, 'a_y_max': a_y_max,
                        'b_x_min': b_x_min, 'b_y_min': b_y_min, 'b_x_max': b_x_max, 'b_y_max': b_y_max,
                        'left_margin': left_margin,
                        'right_margin': right_margin,
                        'bottom_margin': bottom_margin,
                        'top_margin': top_margin,
                        'fully_contained': 1 if fully_contained else 0
                    })
    
    X_list = pos_examples + neg_examples
    y_list = [1] * len(pos_examples) + [0] * len(neg_examples)
    
    X = pd.DataFrame(X_list)
    y = np.array(y_list)
    X.index = [f"e_{i+1}" for i in range(len(X))]
    
    return X, y, bk_file


def load_near_corner_data(examples_file, bk_file=None):
    """Load near_corner data: near_corner(A, B) where A is a point and B is a bounding box"""
    pos_examples = []
    neg_examples = []
    
    def distance_2d(p1, p2):
        return ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)**0.5
    
    with open(examples_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('near_corner(') and not line.startswith('not('):
                match = re.match(r'near_corner\(([^,]+),([^,]+),([^,]+),([^,]+),([^,]+),([^)]+)\)', line.rstrip('.'))
                if match:
                    a_x, a_y = float(match.group(1)), float(match.group(2))
                    b_x_min, b_y_min = float(match.group(3)), float(match.group(4))
                    b_x_max, b_y_max = float(match.group(5)), float(match.group(6))
                    
                    # Calculate distance to all corners
                    corners = [
                        (b_x_min, b_y_min, 'bl'),  # bottom-left
                        (b_x_max, b_y_min, 'br'),  # bottom-right
                        (b_x_min, b_y_max, 'tl'),  # top-left
                        (b_x_max, b_y_max, 'tr')   # top-right
                    ]
                    a = [a_x, a_y]
                    distances = [distance_2d(a, [c[0], c[1]]) for c in corners]
                    min_dist = min(distances)
                    
                    # Distances to each corner
                    dist_bl = distance_2d(a, [corners[0][0], corners[0][1]])
                    dist_br = distance_2d(a, [corners[1][0], corners[1][1]])
                    dist_tl = distance_2d(a, [corners[2][0], corners[2][1]])
                    dist_tr = distance_2d(a, [corners[3][0], corners[3][1]])
                    
                    pos_examples.append({
                        'a_x': a_x, 'a_y': a_y,
                        'b_x_min': b_x_min, 'b_y_min': b_y_min,
                        'b_x_max': b_x_max, 'b_y_max': b_y_max,
                        'min_corner_dist': min_dist,
                        'dist_bl': dist_bl,
                        'dist_br': dist_br,
                        'dist_tl': dist_tl,
                        'dist_tr': dist_tr
                    })
            elif line.startswith('not(near_corner('):
                match = re.match(r'not\(near_corner\(([^,]+),([^,]+),([^,]+),([^,]+),([^,]+),([^)]+)\)\)', line.rstrip('.'))
                if match:
                    a_x, a_y = float(match.group(1)), float(match.group(2))
                    b_x_min, b_y_min = float(match.group(3)), float(match.group(4))
                    b_x_max, b_y_max = float(match.group(5)), float(match.group(6))
                    
                    corners = [
                        (b_x_min, b_y_min), (b_x_max, b_y_min),
                        (b_x_min, b_y_max), (b_x_max, b_y_max)
                    ]
                    a = [a_x, a_y]
                    distances = [distance_2d(a, list(c)) for c in corners]
                    min_dist = min(distances)
                    
                    dist_bl = distance_2d(a, [corners[0][0], corners[0][1]])
                    dist_br = distance_2d(a, [corners[1][0], corners[1][1]])
                    dist_tl = distance_2d(a, [corners[2][0], corners[2][1]])
                    dist_tr = distance_2d(a, [corners[3][0], corners[3][1]])
                    
                    neg_examples.append({
                        'a_x': a_x, 'a_y': a_y,
                        'b_x_min': b_x_min, 'b_y_min': b_y_min,
                        'b_x_max': b_x_max, 'b_y_max': b_y_max,
                        'min_corner_dist': min_dist,
                        'dist_bl': dist_bl,
                        'dist_br': dist_br,
                        'dist_tl': dist_tl,
                        'dist_tr': dist_tr
                    })
    
    X_list = pos_examples + neg_examples
    y_list = [1] * len(pos_examples) + [0] * len(neg_examples)
    
    X = pd.DataFrame(X_list)
    y = np.array(y_list)
    X.index = [f"e_{i+1}" for i in range(len(X))]
    
    return X, y, bk_file

