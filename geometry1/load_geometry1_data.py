#!/usr/bin/env python3
"""
Load geometry1 problems from Prolog files
"""

import re
import pandas as pd
import numpy as np
import os


def load_halfplane3d_data(examples_file, bk_file=None):
    """Load 3D halfplane data"""
    pos_examples = []
    neg_examples = []
    
    with open(examples_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('pos('):
                match = re.match(r'pos\(halfplane3d\(([^,]+),\s*([^,]+),\s*([^)]+)\)\)', line)
                if match:
                    x, y, z = float(match.group(1)), float(match.group(2)), float(match.group(3))
                    pos_examples.append({'x': x, 'y': y, 'z': z})
            elif line.startswith('neg('):
                match = re.match(r'neg\(halfplane3d\(([^,]+),\s*([^,]+),\s*([^)]+)\)\)', line)
                if match:
                    x, y, z = float(match.group(1)), float(match.group(2)), float(match.group(3))
                    neg_examples.append({'x': x, 'y': y, 'z': z})
    
    X_list = pos_examples + neg_examples
    y_list = [1] * len(pos_examples) + [0] * len(neg_examples)
    
    X = pd.DataFrame(X_list)
    y = np.array(y_list)
    X.index = [f"e_{i+1}" for i in range(len(X))]
    
    return X, y, bk_file


def load_conjunction_data(examples_file, bk_file=None):
    """Load conjunction data"""
    pos_examples = []
    neg_examples = []
    
    with open(examples_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('pos('):
                match = re.match(r'pos\(conjunction\(([^,]+),\s*([^,]+),\s*([^)]+)\)\)', line)
                if match:
                    x, y, z = float(match.group(1)), float(match.group(2)), float(match.group(3))
                    pos_examples.append({'x': x, 'y': y, 'z': z})
            elif line.startswith('neg('):
                match = re.match(r'neg\(conjunction\(([^,]+),\s*([^,]+),\s*([^)]+)\)\)', line)
                if match:
                    x, y, z = float(match.group(1)), float(match.group(2)), float(match.group(3))
                    neg_examples.append({'x': x, 'y': y, 'z': z})
    
    X_list = pos_examples + neg_examples
    y_list = [1] * len(pos_examples) + [0] * len(neg_examples)
    
    X = pd.DataFrame(X_list)
    y = np.array(y_list)
    X.index = [f"e_{i+1}" for i in range(len(X))]
    
    return X, y, bk_file


def load_multihalfplane_data(examples_file, bk_file=None):
    """Load multiple halfplanes data"""
    pos_examples = []
    neg_examples = []
    
    with open(examples_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('pos('):
                match = re.match(r'pos\(multihalfplane\(([^,]+),\s*([^)]+)\)\)', line)
                if match:
                    x, y = float(match.group(1)), float(match.group(2))
                    pos_examples.append({'x': x, 'y': y})
            elif line.startswith('neg('):
                match = re.match(r'neg\(multihalfplane\(([^,]+),\s*([^)]+)\)\)', line)
                if match:
                    x, y = float(match.group(1)), float(match.group(2))
                    neg_examples.append({'x': x, 'y': y})
    
    X_list = pos_examples + neg_examples
    y_list = [1] * len(pos_examples) + [0] * len(neg_examples)
    
    X = pd.DataFrame(X_list)
    y = np.array(y_list)
    X.index = [f"e_{i+1}" for i in range(len(X))]
    
    return X, y, bk_file


def load_interval3d_data(examples_file, bk_file=None):
    """Load 3D interval data"""
    pos_examples = []
    neg_examples = []
    
    with open(examples_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('pos('):
                match = re.match(r'pos\(interval3d\(([^,]+),\s*([^,]+),\s*([^)]+)\)\)', line)
                if match:
                    x, y, z = float(match.group(1)), float(match.group(2)), float(match.group(3))
                    pos_examples.append({'x': x, 'y': y, 'z': z})
            elif line.startswith('neg('):
                match = re.match(r'neg\(interval3d\(([^,]+),\s*([^,]+),\s*([^)]+)\)\)', line)
                if match:
                    x, y, z = float(match.group(1)), float(match.group(2)), float(match.group(3))
                    neg_examples.append({'x': x, 'y': y, 'z': z})
    
    X_list = pos_examples + neg_examples
    y_list = [1] * len(pos_examples) + [0] * len(neg_examples)
    
    X = pd.DataFrame(X_list)
    y = np.array(y_list)
    X.index = [f"e_{i+1}" for i in range(len(X))]
    
    return X, y, bk_file

