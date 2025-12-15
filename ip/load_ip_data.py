"""Load InfluencePropagation (IP) data in RELATIONAL format for PyGol"""
import pandas as pd
import re
import os
import math

def load_ip_data(problem_name, data_dir='data'):
    """
    Load IP data in RELATIONAL format (not tabular)
    
    This keeps examples as Prolog atoms (e.g., 'ip1_active(obj3)')
    and preserves propagates/2 facts in BK for PyGol's relational learning.
    
    Args:
        problem_name: One of ['ip1_active', 'ip2_active', 'ip3_active', 'ip3_threshold', 'ip4_high_score']
        data_dir: Directory containing data
        
    Returns:
        X: DataFrame with object IDs (for compatibility) AND numerical features (for Z3)
        y: Labels
        bk_file: Path to BK with propagates facts
    """
    bk_file = os.path.join(data_dir, 'objects_BK.pl')
    examples_file = os.path.join(data_dir, f'{problem_name}_examples.pl')
    
    # Load objects from BK
    objects = {}
    propagates_map = {}  # obj_id -> [target_ids]
    
    with open(bk_file, 'r') as f:
        for line in f:
            if line.startswith('object('):
                # Parse: object(obj1, x, y, score).
                match = re.match(r'object\((\w+),\s*([-\d.]+),\s*([-\d.]+),\s*([-\d.]+)\)', line.strip().rstrip('.'))
                if match:
                    obj_id = match.group(1)
                    x = float(match.group(2))
                    y = float(match.group(3))
                    score = float(match.group(4))
                    objects[obj_id] = {'x': x, 'y': y, 'score': score}
                    propagates_map[obj_id] = []
            elif line.startswith('propagates('):
                # Parse: propagates(obj1, obj2).
                match = re.match(r'propagates\((\w+),\s*(\w+)\)', line.strip().rstrip('.'))
                if match:
                    from_obj = match.group(1)
                    to_obj = match.group(2)
                    if from_obj not in propagates_map:
                        propagates_map[from_obj] = []
                    propagates_map[from_obj].append(to_obj)
    
    # Load examples - keep as RELATIONAL atoms
    pos_examples, neg_examples = [], []
    with open(examples_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('neg('):
                # Parse: neg(ip1_active(obj1)).
                match = re.match(r'neg\((.+)\)', line.rstrip('.'))
                if match:
                    neg_examples.append(match.group(1))  # Keep full atom
            elif line and not line.startswith('%'):
                # Parse: ip1_active(obj1).
                atom = line.rstrip('.')
                if atom:
                    pos_examples.append(atom)  # Keep full atom
    
    # Helper to compute influence
    def influence(obj1_id, obj2_id):
        if obj1_id == obj2_id:
            return 0.0
        obj1 = objects[obj1_id]
        obj2 = objects[obj2_id]
        dist = math.sqrt((obj1['x'] - obj2['x'])**2 + (obj1['y'] - obj2['y'])**2)
        if dist < 0.01:
            return 0.0
        return (obj1['score'] * obj2['score']) / (dist ** 2)
    
    # Create DataFrame with BOTH relational examples AND numerical features
    # This hybrid approach allows:
    # - PyGol to learn from relational structure (propagates facts)
    # - Z3 to optimize numerical thresholds on aggregated features
    
    data = []
    labels = []
    
    for example in pos_examples:
        # Extract object ID from example (e.g., 'ip1_active(obj3)' -> 'obj3')
        match = re.match(r'\w+\((\w+)\)', example)
        if match and match.group(1) in objects:
            obj_id = match.group(1)
            obj = objects[obj_id]
            
            # Compute influence-based features (for Z3 optimization)
            propagates_to = propagates_map.get(obj_id, [])
            num_out = len(propagates_to)
            
            if num_out > 0:
                influences = [influence(obj_id, target) for target in propagates_to]
                max_out_influence = max(influences)
                avg_out_influence = sum(influences) / len(influences)
                total_neighbor_score = sum(objects[t]['score'] for t in propagates_to)
            else:
                max_out_influence = 0.0
                avg_out_influence = 0.0
                total_neighbor_score = 0.0
            
            # Compute max chain score (for ip4_high_score)
            # Find all 3-hop chains from this object and compute max aggregate score
            max_chain_score = obj['score']  # Start with self
            try:
                for b_id in propagates_map.get(obj_id, []):
                    if b_id in objects:
                        for c_id in propagates_map.get(b_id, []):
                            if c_id != obj_id and c_id in objects:
                                for d_id in propagates_map.get(c_id, []):
                                    if d_id not in [obj_id, b_id] and d_id in objects:
                                        # Chain: obj -> b -> c -> d
                                        chain_score = obj['score'] + objects[b_id]['score'] + objects[c_id]['score'] + objects[d_id]['score']
                                        max_chain_score = max(max_chain_score, chain_score)
            except Exception as e:
                # If computation fails, use just self score
                max_chain_score = obj['score']
            
            row = {
                'example': example,  # Keep relational example for PyGol
                'obj_id': obj_id,
                # Basic features
                'x': obj['x'],
                'y': obj['y'],
                'score': obj['score'],
                'dist_from_origin': math.sqrt(obj['x']**2 + obj['y']**2),
                # Propagation features (for Z3)
                'num_propagates_out': num_out,
                'max_out_influence': max_out_influence,
                'avg_out_influence': avg_out_influence,
                'total_neighbor_score': total_neighbor_score,
                'max_chain_score': max_chain_score  # For ip4_high_score
            }
            data.append(row)
            labels.append(1)
    
    for example in neg_examples:
        # Extract object ID
        match = re.match(r'\w+\((\w+)\)', example)
        if match and match.group(1) in objects:
            obj_id = match.group(1)
            obj = objects[obj_id]
            
            # Compute same features
            propagates_to = propagates_map.get(obj_id, [])
            num_out = len(propagates_to)
            
            if num_out > 0:
                influences = [influence(obj_id, target) for target in propagates_to]
                max_out_influence = max(influences)
                avg_out_influence = sum(influences) / len(influences)
                total_neighbor_score = sum(objects[t]['score'] for t in propagates_to)
            else:
                max_out_influence = 0.0
                avg_out_influence = 0.0
                total_neighbor_score = 0.0
            
            # Compute max chain score (for ip4_high_score)
            max_chain_score = obj['score']
            try:
                for b_id in propagates_map.get(obj_id, []):
                    if b_id in objects:
                        for c_id in propagates_map.get(b_id, []):
                            if c_id != obj_id and c_id in objects:
                                for d_id in propagates_map.get(c_id, []):
                                    if d_id not in [obj_id, b_id] and d_id in objects:
                                        chain_score = obj['score'] + objects[b_id]['score'] + objects[c_id]['score'] + objects[d_id]['score']
                                        max_chain_score = max(max_chain_score, chain_score)
            except Exception as e:
                max_chain_score = obj['score']
            
            row = {
                'example': example,  # Keep relational example for PyGol
                'obj_id': obj_id,
                # Basic features
                'x': obj['x'],
                'y': obj['y'],
                'score': obj['score'],
                'dist_from_origin': math.sqrt(obj['x']**2 + obj['y']**2),
                # Propagation features (for Z3)
                'num_propagates_out': num_out,
                'max_out_influence': max_out_influence,
                'avg_out_influence': avg_out_influence,
                'total_neighbor_score': total_neighbor_score,
                'max_chain_score': max_chain_score
            }
            data.append(row)
            labels.append(0)
    
    X = pd.DataFrame(data)
    y = pd.Series(labels)
    
    # Return absolute path to avoid issues when working directory changes
    bk_file_abs = os.path.abspath(bk_file)
    
    return X, y, bk_file_abs
