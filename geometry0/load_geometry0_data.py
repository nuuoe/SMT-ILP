#!/usr/bin/env python3
"""
Load geometry0 problems from Prolog files
"""

import re
import pandas as pd
import numpy as np
import os


def load_interval_data(examples_file, bk_file=None):
    """Load interval data from examples file (bk_file is optional, not used)"""
    """Load interval data"""
    pos_examples = []
    neg_examples = []
    
    with open(examples_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('pos('):
                match = re.match(r'pos\(interval\(([^)]+)\)\)', line)
                if match:
                    x = float(match.group(1))
                    pos_examples.append({'x': x})
            elif line.startswith('neg('):
                match = re.match(r'neg\(interval\(([^)]+)\)\)', line)
                if match:
                    x = float(match.group(1))
                    neg_examples.append({'x': x})
    
    X_list = pos_examples + neg_examples
    y_list = [1] * len(pos_examples) + [0] * len(neg_examples)
    
    X = pd.DataFrame(X_list)
    y = np.array(y_list)
    X.index = [f"e_{i+1}" for i in range(len(X))]
    
    return X, y, bk_file


def load_halfplane_data(examples_file, bk_file=None):
    """Load halfplane data from examples file (bk_file is optional, not used)"""
    """Load halfplane data"""
    pos_examples = []
    neg_examples = []
    
    with open(examples_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('pos('):
                match = re.match(r'pos\(halfplane\(([^,]+),\s*([^)]+)\)\)', line)
                if match:
                    x, y = float(match.group(1)), float(match.group(2))
                    pos_examples.append({'x': x, 'y': y})
            elif line.startswith('neg('):
                match = re.match(r'neg\(halfplane\(([^,]+),\s*([^)]+)\)\)', line)
                if match:
                    x, y = float(match.group(1)), float(match.group(2))
                    neg_examples.append({'x': x, 'y': y})
    
    X_list = pos_examples + neg_examples
    y_list = [1] * len(pos_examples) + [0] * len(neg_examples)
    
    X = pd.DataFrame(X_list)
    y = np.array(y_list)
    X.index = [f"e_{i+1}" for i in range(len(X))]
    
    return X, y, bk_file

