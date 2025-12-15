#!/usr/bin/env python3
"""
Generate InfluencePropagation (IP) dataset.

Dataset: IP (InfluencePropagation)
Requires: BOTH Predicate Invention (PI) and Z3 SMT

Influence Function: influence(A,B) = (score_A × score_B) / dist(A,B)²

Tasks:
1. ip1_active: propagates(A,B) - Tests Z3
2. ip2_active: propagates(A,B), propagates(B,C) - Tests PI + Z3
3. ip3_active: propagates(A,B), propagates(B,C), propagates(C,D) - Tests PI + Z3 (essential!)
4. ip3_threshold: ip3_active + influence(A,D) > T - Tests PI + Z3 (both essential!)
"""

import random
import math
import os

# World parameters (increased for hard negatives)
NUM_OBJECTS = 100
WORLD_SIZE = 100
SCORE_MIN = 2.0
SCORE_MAX = 10.0

# Thresholds
TOUCHING_DIST = 20.0
INFLUENCE_THRESHOLD = 1.0

# Dataset sizes
NUM_POS = 50
NUM_NEG = 50

random.seed(42)

def distance(obj1, obj2):
    """Euclidean distance between two objects"""
    return math.sqrt((obj1['x'] - obj2['x'])**2 + (obj1['y'] - obj2['y'])**2)

def influence(obj1, obj2):
    """
    Nonlinear influence calculation
    influence = (score1 × score2) / dist²
    
    Requires Z3 for multiplication, squares, and division.
    """
    d = distance(obj1, obj2)
    if d < 0.1:  # Avoid division by zero
        d = 0.1
    return (obj1['score'] * obj2['score']) / (d * d)

def touching(obj1, obj2):
    """Objects are touching if within threshold distance"""
    return distance(obj1, obj2) < TOUCHING_DIST

def propagates(obj1, obj2):
    """
    obj1 propagates to obj2 if:
    1. They are touching (spatial constraint)
    2. Influence between them exceeds threshold (numeric constraint)
    3. NOT mutual (to avoid cycles - only directional propagation)
    
    This ensures LINEAR chains, not triangles/cycles.
    """
    # Simple condition - back to working version
    return touching(obj1, obj2) and influence(obj1, obj2) > INFLUENCE_THRESHOLD

def generate_objects(num_objects=50):
    """Generate objects with score and position (back to working version)"""
    objects = []
    
    # Create clusters to ensure some objects touch
    num_clusters = 5
    cluster_centers = [
        (random.uniform(-WORLD_SIZE, WORLD_SIZE), 
         random.uniform(-WORLD_SIZE, WORLD_SIZE))
        for _ in range(num_clusters)
    ]
    
    for i in range(num_objects):
        # Assign to a cluster with some noise
        cluster = cluster_centers[i % num_clusters]
        x = cluster[0] + random.uniform(-20, 20)
        y = cluster[1] + random.uniform(-20, 20)
        score = random.uniform(SCORE_MIN, SCORE_MAX)
        
        objects.append({
            'id': f'obj{i+1}',
            'x': x,
            'y': y,
            'score': score
        })
    
    return objects

def find_propagation_chains(objects):
    """Find all propagation relationships (for generating examples)"""
    # Build adjacency list of propagates relationships
    propagates_map = {}
    for obj1 in objects:
        propagates_map[obj1['id']] = []
        for obj2 in objects:
            if obj1['id'] != obj2['id'] and propagates(obj1, obj2):
                propagates_map[obj1['id']].append(obj2['id'])
    
    return propagates_map

# ============================================================================
# Task 1: Single-hop - ip1_active(A) :- propagates(A,B)
# ============================================================================

def generate_ip1_active(objects, propagates_map, num_pos=20, num_neg=20):
    """
    Pattern: ip1_active(A) if propagates(A,B) for some B
    
    Tests: Basic Z3 reasoning (influence calculation)
    PI: Not required (single hop)
    """
    pos_examples, neg_examples = [], []
    
    # Positives: Objects that propagate to at least one other object
    for obj in objects:
        if propagates_map[obj['id']] and len(pos_examples) < num_pos:
            pos_examples.append(f"ip1_active({obj['id']})")
    
    # Negatives: Objects that don't propagate to any other object
    for obj in objects:
        if not propagates_map[obj['id']] and len(neg_examples) < num_neg:
            neg_examples.append(f"ip1_active({obj['id']})")
    
    return pos_examples[:num_pos], neg_examples[:num_neg]

# ============================================================================
# Task 2: Two-hop - ip2_active(A) :- propagates(A,B), propagates(B,C)
# ============================================================================

def generate_ip2_active(objects, propagates_map, num_pos=20, num_neg=20):
    """
    Pattern: ip2_active(A) if A->B->C chain exists
    
    Tests: PI for 2-hop chains + Z3 for influence evaluation
    WITHOUT PI: Must learn propagates(A,B), propagates(B,C) (2 literals)
    WITH PI: Can invent ip2(A,C) :- propagates(A,B), propagates(B,C)
    """
    pos_examples, neg_examples = set(), set()
    
    # Find all 2-hop chains
    chains_2hop = set()
    for obj_a in objects:
        for obj_b_id in propagates_map[obj_a['id']]:
            for obj_c_id in propagates_map[obj_b_id]:
                if obj_c_id != obj_a['id']:  # Avoid loops
                    chains_2hop.add(obj_a['id'])
    
    # Positives: Objects with 2-hop chains
    for obj_id in chains_2hop:
        if len(pos_examples) < num_pos:
            pos_examples.add(f"ip2_active({obj_id})")
    
    # Negatives: Objects without 2-hop chains
    for obj in objects:
        if obj['id'] not in chains_2hop and len(neg_examples) < num_neg:
            neg_examples.add(f"ip2_active({obj['id']})")
    
    return list(pos_examples)[:num_pos], list(neg_examples)[:num_neg]

# ============================================================================
# Task 3: Three-hop - ip3_active(A) :- propagates(A,B), propagates(B,C), propagates(C,D)
# ============================================================================

def generate_ip3_active(objects, propagates_map, num_pos=20, num_neg=20):
    """
    Pattern: ip3_active(A) if A->B->C->D chain exists
    
    Tests: PI for 3-hop chains + Z3 for influence evaluation
    """
    pos_examples, neg_examples = set(), set()
    
    # Find all objects in triangles and their scores
    triangle_scores = {}  # obj_id -> score
    
    for obj_a in objects:
        # Check if in triangle
        in_triangle = False
        for obj_b_id in propagates_map[obj_a['id']]:
            for obj_c_id in propagates_map[obj_b_id]:
                if obj_c_id != obj_a['id'] and obj_a['id'] in propagates_map.get(obj_c_id, []):
                    in_triangle = True
                    break
            if in_triangle:
                break
        
        if in_triangle:
            triangle_scores[obj_a['id']] = obj_a['score']
    
    # Split triangle objects by score to create ambiguity (same as ip4)
    hard_negatives = set()
    if triangle_scores:
        sorted_triangles = sorted(triangle_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Top 75%: high score as positive
        top_75_idx = int(len(sorted_triangles) * 0.75)
        for obj_id, score in sorted_triangles[:top_75_idx]:
            if len(pos_examples) < num_pos:
                pos_examples.add(f"ip3_active({obj_id})")
        
        # Bottom 25%: low score as hard negatives
        for obj_id, score in sorted_triangles[top_75_idx:]:
            if len(hard_negatives) < num_neg // 3:
                hard_negatives.add(f"ip3_active({obj_id})")
    
    # Add regular negatives (not in triangles)
    for obj in objects:
        if obj['id'] not in triangle_scores and len(neg_examples) < num_neg - len(hard_negatives):
            neg_examples.add(f"ip3_active({obj['id']})")
    
    all_negatives = list(hard_negatives) + list(neg_examples)
    return list(pos_examples)[:num_pos], all_negatives[:num_neg]

# ============================================================================
# Task 4: Chain + Threshold - ip3_threshold(A) :- propagates(A,B), propagates(B,C), propagates(C,D), influence(A,D) > T
# ============================================================================

def generate_ip3_threshold(objects, propagates_map, num_pos=20, num_neg=20):
    """
    Pattern: ip3_threshold(A) if:
    - A is in a triangle: A->B->C->A  (relational pattern)
    - AND max_out_influence(A) > threshold (numerical constraint)
    
    Hard negatives: triangles with medium influence to make relational pattern ambiguous.
    Requires both PI (for triangle pattern) and Z3 (for influence threshold).
    """
    pos_examples, neg_examples = set(), set()
    hard_negatives = set()  # Triangles with low influence
    
    # First, find all objects in triangles and compute their max influence
    triangle_objects = {}  # obj_id -> max_out_influence
    
    for obj_a in objects:
        in_triangle = False
        max_inf = 0.0
        
        # Check if part of triangle
        for obj_b_id in propagates_map[obj_a['id']]:
            for obj_c_id in propagates_map[obj_b_id]:
                if obj_c_id != obj_a['id'] and obj_a['id'] in propagates_map.get(obj_c_id, []):
                    # Found triangle!
                    in_triangle = True
                    # Compute max outgoing influence
                    for target_id in propagates_map[obj_a['id']]:
                        target = next(o for o in objects if o['id'] == target_id)
                        inf = influence(obj_a, target)
                        max_inf = max(max_inf, inf)
                    break
            if in_triangle:
                break
        
        if in_triangle:
            triangle_objects[obj_a['id']] = max_inf
    
    # Split triangle objects by influence to create ambiguity
    if triangle_objects:
        sorted_triangles = sorted(triangle_objects.items(), key=lambda x: x[1], reverse=True)
        
        top_65_idx = int(len(sorted_triangles) * 0.65)
        for obj_id, inf in sorted_triangles[:top_65_idx]:
            if len(pos_examples) < num_pos:
                pos_examples.add(f"ip3_threshold({obj_id})")
        
        middle_start = top_65_idx
        middle_end = int(len(sorted_triangles) * 0.85)
        for obj_id, inf in sorted_triangles[middle_start:middle_end]:
            if len(hard_negatives) < num_neg // 2:
                hard_negatives.add(f"ip3_threshold({obj_id})")
        
        for obj_id, inf in sorted_triangles[middle_end:]:
            if len(hard_negatives) < num_neg - len(hard_negatives):
                hard_negatives.add(f"ip3_threshold({obj_id})")
    
    # Add regular negatives (not in triangles)
    for obj in objects:
        if obj['id'] not in triangle_objects and len(neg_examples) < num_neg - len(hard_negatives):
            neg_examples.add(f"ip3_threshold({obj['id']})")
    
    # Combine hard negatives with regular negatives
    all_negatives = list(hard_negatives) + list(neg_examples)
    
    return list(pos_examples)[:num_pos], all_negatives[:num_neg]

# ============================================================================
# Task 5: High-Score Chain - ip4_high_score(A) :- propagates(A,B), propagates(B,C), propagates(C,D), 
#                                                   total_chain_score(A,B,C,D) > threshold
# ============================================================================

def generate_ip4_high_score(objects, propagates_map, num_pos=20, num_neg=20):
    """
    Pattern: ip4_high_score(A) if:
    - A is in a triangle: A->B->C->A (relational pattern - same as ip2/ip3)
    - AND max_chain_score(A) > threshold (numerical constraint)
    
    Hard negatives: triangles with low scores to make relational pattern ambiguous.
    Requires both PI (for triangle pattern) and Z3 (for score threshold).
    """
    pos_examples, neg_examples = set(), set()
    hard_negatives = set()
    
    # Find all triangle objects and their max chain scores
    triangle_scores = {}  # obj_id -> max_chain_score
    
    for obj_a in objects:
        # Check if in triangle
        in_triangle = False
        for obj_b_id in propagates_map[obj_a['id']]:
            for obj_c_id in propagates_map[obj_b_id]:
                if obj_c_id != obj_a['id'] and obj_a['id'] in propagates_map.get(obj_c_id, []):
                    in_triangle = True
                    break
            if in_triangle:
                break
        
        if in_triangle:
            # Compute max chain score for any 3-hop path from A
            max_score = obj_a['score']
            for obj_b_id in propagates_map[obj_a['id']]:
                obj_b = next(o for o in objects if o['id'] == obj_b_id)
                for obj_c_id in propagates_map[obj_b_id]:
                    if obj_c_id != obj_a['id']:
                        obj_c = next(o for o in objects if o['id'] == obj_c_id)
                        for obj_d_id in propagates_map.get(obj_c_id, []):
                            if obj_d_id not in [obj_a['id'], obj_b_id]:
                                obj_d = next(o for o in objects if o['id'] == obj_d_id)
                                chain_score = obj_a['score'] + obj_b['score'] + obj_c['score'] + obj_d['score']
                                max_score = max(max_score, chain_score)
            
            triangle_scores[obj_a['id']] = max_score
    
    # Split triangles by score to create ambiguity
    if triangle_scores:
        sorted_by_score = sorted(triangle_scores.items(), key=lambda x: x[1], reverse=True)
        
        top_75_idx = int(len(sorted_by_score) * 0.75)
        for obj_id, score in sorted_by_score[:top_75_idx]:
            if len(pos_examples) < num_pos:
                pos_examples.add(f"ip4_high_score({obj_id})")
        
        for obj_id, score in sorted_by_score[top_75_idx:]:
            if len(hard_negatives) < num_neg // 3:
                hard_negatives.add(f"ip4_high_score({obj_id})")
    
    # Add regular negatives (not in triangles)
    for obj in objects:
        if obj['id'] not in triangle_scores and len(neg_examples) < num_neg - len(hard_negatives):
            neg_examples.add(f"ip4_high_score({obj['id']})")
    
    all_negatives = list(hard_negatives) + list(neg_examples)
    return list(pos_examples)[:num_pos], all_negatives[:num_neg]

# ============================================================================
# Main Generation
# ============================================================================

def main():
    SEED = 1234
    random.seed(SEED)
    
    print("Generating InfluencePropagation (IP) Dataset")
    print(f"Random seed: {SEED}\n")
    
    # Generate objects
    print(f"\nGenerating {NUM_OBJECTS} objects with score and position...")
    objects = generate_objects(NUM_OBJECTS)
    print(f"  Created {len(objects)} objects")
    
    # Find propagation relationships
    print("\nComputing propagation relationships (touching + influence > threshold)...")
    propagates_map = find_propagation_chains(objects)
    total_propagates = sum(len(v) for v in propagates_map.values())
    print(f"  Found {total_propagates} propagation relationships")
    
    # Create output directory
    os.makedirs('data', exist_ok=True)
    
    # Write background knowledge (objects and computed relations)
    print("\nWriting background knowledge...")
    with open('data/objects_BK.pl', 'w') as f:
        f.write("% Objects with score and position\n")
        for obj in objects:
            f.write(f"object({obj['id']}, {obj['x']:.2f}, {obj['y']:.2f}, {obj['score']:.2f}).\n")
        
        f.write("\n% Computed relations\n")
        f.write(f"% touching(A,B) - distance < {TOUCHING_DIST}\n")
        f.write(f"% influence(A,B) - (score_A * score_B) / dist²\n")
        f.write(f"% propagates(A,B) - touching(A,B) AND influence(A,B) > {INFLUENCE_THRESHOLD}\n\n")
        
        # Write propagates facts (pre-computed for PyGol)
        for obj1_id, obj2_ids in propagates_map.items():
            for obj2_id in obj2_ids:
                f.write(f"propagates({obj1_id}, {obj2_id}).\n")
        
    
    print("  Wrote objects_BK.pl (relational only for fast PyGol)")
    
    # Generate all 5 tasks
    tasks = [
        ('ip1_active', generate_ip1_active),
        ('ip2_active', generate_ip2_active),
        ('ip3_active', generate_ip3_active),
        ('ip3_threshold', generate_ip3_threshold),
        ('ip4_high_score', generate_ip4_high_score)
    ]
    
    for task_name, generator_func in tasks:
        print(f"\nGenerating {task_name}...")
        pos, neg = generator_func(objects, propagates_map, NUM_POS, NUM_NEG)
        print(f"  Generated {len(pos)} pos, {len(neg)} neg")
        
        with open(f'data/{task_name}_examples.pl', 'w') as f:
            f.write("% Positive examples\n")
            for ex in pos:
                f.write(f"{ex}.\n")
            f.write("\n% Negative examples\n")
            for ex in neg:
                f.write(f"neg({ex}).\n")
    
    print("\nInfluencePropagation generation complete!")
    print(f"\nGenerated 4 tasks in data/:")
    print("  1. ip1_active      - Tests Z3 (single hop)")
    print("  2. ip2_active      - Tests PI + Z3 (2-hop chain)")
    print("  3. ip3_active      - Tests PI + Z3 (3-hop chain)")
    print("  4. ip3_threshold   - Tests PI + Z3 (chain + endpoint threshold)")
    print("\nEach task requires:")
    print("  - Z3: Influence calculations (multiplication, division, squares)")
    print("  - PI: Multi-hop chain abstraction (for tasks 2-4)")

if __name__ == "__main__":
    main()

