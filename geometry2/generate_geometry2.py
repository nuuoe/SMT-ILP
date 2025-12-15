#!/usr/bin/env python3
"""
Generate Geometry2 problems: Relational Spatial Constraints
1. left_of(A, B): A is to the left of B
2. closer_than(A, B, C): A is closer to B than C is to B
3. touching(A, B): A and B are touching (within threshold distance)
4. inside(A, B): A is inside B (A's bounding box is within B's bounding box)
5. overlapping(A, B): Two bounding boxes overlap
6. between(A, B, C): A is between B and C (collinear and between)
7. adjacent(A, B): Two bounding boxes are adjacent (touching at edges)
8. aligned(A, B, C): Three points are collinear
9. surrounds(A, B): Bounding box A surrounds bounding box B
10. near_corner(A, B): Point A is near a corner of bounding box B
"""

import random
import os

WORLD_SIZE = 100
PRECISION = 0.1
TOUCHING_THRESHOLD = 5.0  # Distance threshold for "touching"
ADJACENT_THRESHOLD = 3.0  # Distance threshold for "adjacent"
NEAR_CORNER_THRESHOLD = 5.0  # Distance threshold for "near corner"

def gen_point_2d():
    """Generate a random 2D point"""
    return [random.randint(-WORLD_SIZE, WORLD_SIZE) for _ in range(2)]

def gen_point_3d():
    """Generate a random 3D point"""
    return [random.randint(-WORLD_SIZE, WORLD_SIZE) for _ in range(3)]

def distance_2d(p1, p2):
    """Calculate Euclidean distance between two 2D points"""
    return ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)**0.5

def distance_3d(p1, p2):
    """Calculate Euclidean distance between two 3D points"""
    return ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2 + (p1[2] - p2[2])**2)**0.5

def point_to_string(p):
    """Convert point to string representation"""
    if len(p) == 2:
        return f"{p[0]},{p[1]}"
    else:
        return f"{p[0]},{p[1]},{p[2]}"

def safe_randint(min_val, max_val):
    """Safely generate random integer with bounds checking"""
    if min_val > max_val:
        return min_val  # Return minimum if range is invalid
    return random.randint(min_val, max_val)


# Problem 1: left_of
def generate_left_of(num_pos=30, num_neg=30):
    """Generate left_of problem: A is to the left of B (A.x < B.x)"""
    pos_examples = []
    neg_examples = []
    
    # Generate positive examples: A.x < B.x
    while len(pos_examples) < num_pos:
        a = gen_point_2d()
        b = gen_point_2d()
        if a[0] < b[0] - PRECISION:  # A is clearly to the left
            pos_examples.append(f'left_of({point_to_string(a)},{point_to_string(b)})')
    
    # Generate negative examples: A.x >= B.x or other invalid cases
    while len(neg_examples) < num_neg:
        a = gen_point_2d()
        b = gen_point_2d()
        if a[0] >= b[0] + PRECISION:  # A is not to the left
            neg_examples.append(f'left_of({point_to_string(a)},{point_to_string(b)})')
    
    return pos_examples, neg_examples, {}


# Problem 2: closer_than
def generate_closer_than(num_pos=30, num_neg=30):
    """Generate closer_than problem: A is closer to B than C is to B"""
    pos_examples = []
    neg_examples = []
    
    # Generate positive examples: dist(A,B) < dist(C,B)
    while len(pos_examples) < num_pos:
        a = gen_point_2d()
        b = gen_point_2d()
        c = gen_point_2d()
        dist_ab = distance_2d(a, b)
        dist_cb = distance_2d(c, b)
        if dist_ab < dist_cb - PRECISION:
            pos_examples.append(f'closer_than({point_to_string(a)},{point_to_string(b)},{point_to_string(c)})')
    
    # Generate negative examples: dist(A,B) >= dist(C,B)
    while len(neg_examples) < num_neg:
        a = gen_point_2d()
        b = gen_point_2d()
        c = gen_point_2d()
        dist_ab = distance_2d(a, b)
        dist_cb = distance_2d(c, b)
        if dist_ab >= dist_cb + PRECISION:
            neg_examples.append(f'closer_than({point_to_string(a)},{point_to_string(b)},{point_to_string(c)})')
    
    return pos_examples, neg_examples, {}


# Problem 3: touching
def generate_touching(num_pos=30, num_neg=30):
    """Generate touching problem: A and B are touching (within threshold distance)"""
    pos_examples = []
    neg_examples = []
    
    # Generate positive examples: dist(A,B) <= TOUCHING_THRESHOLD
    while len(pos_examples) < num_pos:
        a = gen_point_2d()
        b = gen_point_2d()
        dist = distance_2d(a, b)
        if dist <= TOUCHING_THRESHOLD - PRECISION:
            pos_examples.append(f'touching({point_to_string(a)},{point_to_string(b)})')
    
    # Generate negative examples: dist(A,B) > TOUCHING_THRESHOLD
    while len(neg_examples) < num_neg:
        a = gen_point_2d()
        b = gen_point_2d()
        dist = distance_2d(a, b)
        if dist > TOUCHING_THRESHOLD + PRECISION:
            neg_examples.append(f'touching({point_to_string(a)},{point_to_string(b)})')
    
    return pos_examples, neg_examples, {'threshold': TOUCHING_THRESHOLD}


# Problem 4: inside
def generate_inside(num_pos=30, num_neg=30):
    """Generate inside problem: A is inside B (A's bounding box is within B's bounding box)"""
    pos_examples = []
    neg_examples = []
    
    # Generate positive examples: A is inside B's bounding box
    # B is represented as a bounding box (x_min, y_min, x_max, y_max)
    # A is a point (x, y)
    while len(pos_examples) < num_pos:
        # Generate B's bounding box
        b_x_min = random.randint(-WORLD_SIZE//2, WORLD_SIZE//4)
        b_x_max = random.randint(b_x_min + 10, WORLD_SIZE//2)
        b_y_min = random.randint(-WORLD_SIZE//2, WORLD_SIZE//4)
        b_y_max = random.randint(b_y_min + 10, WORLD_SIZE//2)
        
        # Generate A inside B
        a_x = random.randint(b_x_min + 1, b_x_max - 1)
        a_y = random.randint(b_y_min + 1, b_y_max - 1)
        a = [a_x, a_y]
        b = [b_x_min, b_y_min, b_x_max, b_y_max]  # Bounding box
        
        pos_examples.append(f'inside({a[0]},{a[1]},{b[0]},{b[1]},{b[2]},{b[3]})')
    
    # Generate negative examples: A is outside B's bounding box
    while len(neg_examples) < num_neg:
        # Generate B's bounding box
        b_x_min = random.randint(-WORLD_SIZE//2, WORLD_SIZE//4)
        b_x_max = random.randint(b_x_min + 10, WORLD_SIZE//2)
        b_y_min = random.randint(-WORLD_SIZE//2, WORLD_SIZE//4)
        b_y_max = random.randint(b_y_min + 10, WORLD_SIZE//2)
        
        # Generate A outside B (either x or y is outside)
        if random.random() < 0.5:
            # A.x is outside
            a_x = random.choice([random.randint(-WORLD_SIZE, b_x_min - 1), 
                                random.randint(b_x_max + 1, WORLD_SIZE)])
            a_y = random.randint(b_y_min, b_y_max)
        else:
            # A.y is outside
            a_x = random.randint(b_x_min, b_x_max)
            a_y = random.choice([random.randint(-WORLD_SIZE, b_y_min - 1),
                                random.randint(b_y_max + 1, WORLD_SIZE)])
        a = [a_x, a_y]
        b = [b_x_min, b_y_min, b_x_max, b_y_max]
        
        neg_examples.append(f'inside({a[0]},{a[1]},{b[0]},{b[1]},{b[2]},{b[3]})')
    
    return pos_examples, neg_examples, {}


# Problem 5: overlapping
def generate_overlapping(num_pos=30, num_neg=30):
    """Generate overlapping problem: Two bounding boxes overlap"""
    pos_examples = []
    neg_examples = []
    
    def boxes_overlap(box1, box2):
        """Check if two bounding boxes overlap"""
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2
        # Boxes overlap if they overlap in both dimensions
        x_overlap = not (x1_max < x2_min or x2_max < x1_min)
        y_overlap = not (y1_max < y2_min or y2_max < y1_min)
        return x_overlap and y_overlap
    
    # Generate positive examples: boxes overlap
    while len(pos_examples) < num_pos:
        # Generate first bounding box
        a_x_min = random.randint(-WORLD_SIZE//2, WORLD_SIZE//4)
        a_x_max = random.randint(a_x_min + 5, WORLD_SIZE//2)
        a_y_min = random.randint(-WORLD_SIZE//2, WORLD_SIZE//4)
        a_y_max = random.randint(a_y_min + 5, WORLD_SIZE//2)
        box_a = [a_x_min, a_y_min, a_x_max, a_y_max]
        
        # Generate second bounding box that overlaps with first
        # Ensure valid ranges for b_x_min
        b_x_min_low = max(-WORLD_SIZE//2, a_x_min - 10)
        b_x_min_high = max(b_x_min_low, min(a_x_max - 2, WORLD_SIZE//2 - 10))
        b_x_min = random.randint(b_x_min_low, b_x_min_high) if b_x_min_low <= b_x_min_high else b_x_min_low
        
        # Ensure b_x_max > b_x_min
        b_x_max_low = max(a_x_min + 2, b_x_min + 5)
        b_x_max_high = min(a_x_max + 10, WORLD_SIZE//2)
        b_x_max = random.randint(b_x_max_low, b_x_max_high) if b_x_max_low <= b_x_max_high else b_x_max_low + 5
        
        # Similar for y coordinates
        b_y_min_low = max(-WORLD_SIZE//2, a_y_min - 10)
        b_y_min_high = max(b_y_min_low, min(a_y_max - 2, WORLD_SIZE//2 - 10))
        b_y_min = random.randint(b_y_min_low, b_y_min_high) if b_y_min_low <= b_y_min_high else b_y_min_low
        
        b_y_max_low = max(a_y_min + 2, b_y_min + 5)
        b_y_max_high = min(a_y_max + 10, WORLD_SIZE//2)
        b_y_max = random.randint(b_y_max_low, b_y_max_high) if b_y_max_low <= b_y_max_high else b_y_max_low + 5
        box_b = [b_x_min, b_y_min, b_x_max, b_y_max]
        
        if boxes_overlap(box_a, box_b):
            pos_examples.append(f'overlapping({a_x_min},{a_y_min},{a_x_max},{a_y_max},{b_x_min},{b_y_min},{b_x_max},{b_y_max})')
    
    # Generate negative examples: boxes don't overlap
    while len(neg_examples) < num_neg:
        # Generate first bounding box
        a_x_min = random.randint(-WORLD_SIZE//2, WORLD_SIZE//4)
        a_x_max = random.randint(a_x_min + 5, WORLD_SIZE//2)
        a_y_min = random.randint(-WORLD_SIZE//2, WORLD_SIZE//4)
        a_y_max = random.randint(a_y_min + 5, WORLD_SIZE//2)
        box_a = [a_x_min, a_y_min, a_x_max, a_y_max]
        
        # Generate second bounding box that doesn't overlap
        if random.random() < 0.5:
            # B is to the right of A
            b_x_min = random.randint(a_x_max + 1, WORLD_SIZE - 10)
            b_x_max = random.randint(b_x_min + 5, WORLD_SIZE)
            b_y_min = random.randint(-WORLD_SIZE//2, WORLD_SIZE//2 - 10)
            b_y_max = random.randint(b_y_min + 5, WORLD_SIZE//2)
        else:
            # B is above A
            b_x_min = random.randint(-WORLD_SIZE//2, WORLD_SIZE//2 - 10)
            b_x_max = random.randint(b_x_min + 5, WORLD_SIZE//2)
            b_y_min = random.randint(a_y_max + 1, WORLD_SIZE - 10)
            b_y_max = random.randint(b_y_min + 5, WORLD_SIZE)
        box_b = [b_x_min, b_y_min, b_x_max, b_y_max]
        
        if not boxes_overlap(box_a, box_b):
            neg_examples.append(f'overlapping({a_x_min},{a_y_min},{a_x_max},{a_y_max},{b_x_min},{b_y_min},{b_x_max},{b_y_max})')
    
    return pos_examples, neg_examples, {}


# Problem 6: between
def generate_between(num_pos=30, num_neg=30):
    """Generate between problem: A is between B and C (collinear and between)"""
    pos_examples = []
    neg_examples = []
    
    def point_between(p, q, r):
        """Check if point q is between points p and r (collinear and between)"""
        # Check if collinear (using cross product)
        cross = (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])
        if abs(cross) > PRECISION:  # Not collinear
            return False
        # Check if q is between p and r (using dot product)
        # q is between if (q - p) dot (r - q) >= 0
        dx1 = q[0] - p[0]
        dy1 = q[1] - p[1]
        dx2 = r[0] - q[0]
        dy2 = r[1] - q[1]
        dot = dx1 * dx2 + dy1 * dy2
        return dot >= -PRECISION
    
    # Generate positive examples: A is between B and C
    while len(pos_examples) < num_pos:
        # Generate B and C
        b = gen_point_2d()
        c = gen_point_2d()
        # Generate A on the line segment between B and C
        t = random.uniform(0.2, 0.8)  # Random point between 20% and 80% along segment
        a_x = b[0] + t * (c[0] - b[0])
        a_y = b[1] + t * (c[1] - b[1])
        a = [a_x, a_y]
        
        if point_between(b, a, c):
            pos_examples.append(f'between({point_to_string(a)},{point_to_string(b)},{point_to_string(c)})')
    
    # Generate negative examples: A is not between B and C
    while len(neg_examples) < num_neg:
        b = gen_point_2d()
        c = gen_point_2d()
        # Generate A not on the line segment
        if random.random() < 0.5:
            # A is collinear but outside the segment (beyond C)
            t = random.uniform(1.5, 2.5)
            a_x = b[0] + t * (c[0] - b[0])
            a_y = b[1] + t * (c[1] - b[1])
        else:
            # A is not collinear at all
            a = gen_point_2d()
            a_x, a_y = a[0], a[1]
        a = [a_x, a_y]
        
        if not point_between(b, a, c):
            neg_examples.append(f'between({point_to_string(a)},{point_to_string(b)},{point_to_string(c)})')
    
    return pos_examples, neg_examples, {}


# Problem 7: adjacent
def generate_adjacent(num_pos=50, num_neg=50):
    """Generate adjacent problem: Two bounding boxes are adjacent (touching at edges)"""
    pos_examples = []
    neg_examples = []
    
    def boxes_adjacent(box1, box2):
        """Check if two bounding boxes are adjacent (touching at edges)"""
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2
        
        # Check if boxes overlap (if so, not adjacent)
        if not (x1_max < x2_min or x2_max < x1_min) and not (y1_max < y2_min or y2_max < y1_min):
            return False
        
        # Check if they're touching: adjacent horizontally or vertically
        # Horizontally adjacent: one box's right edge touches other's left edge (or vice versa)
        horizontal_adjacent = ((abs(x1_max - x2_min) < ADJACENT_THRESHOLD or abs(x2_max - x1_min) < ADJACENT_THRESHOLD) and
                               not (y1_max < y2_min or y2_max < y1_min))
        # Vertically adjacent: one box's top edge touches other's bottom edge (or vice versa)
        vertical_adjacent = ((abs(y1_max - y2_min) < ADJACENT_THRESHOLD or abs(y2_max - y1_min) < ADJACENT_THRESHOLD) and
                             not (x1_max < x2_min or x2_max < x1_min))
        
        return horizontal_adjacent or vertical_adjacent
    
    # Generate positive examples: boxes are adjacent
    while len(pos_examples) < num_pos:
        # Generate first bounding box
        a_x_min = random.randint(-WORLD_SIZE//2, WORLD_SIZE//4)
        a_x_max = random.randint(a_x_min + 5, WORLD_SIZE//2)
        a_y_min = random.randint(-WORLD_SIZE//2, WORLD_SIZE//4)
        a_y_max = random.randint(a_y_min + 5, WORLD_SIZE//2)
        box_a = [a_x_min, a_y_min, a_x_max, a_y_max]
        
        # Generate second bounding box adjacent to first
        if random.random() < 0.5:
            # B is to the right of A (touching)
            gap = random.uniform(0, ADJACENT_THRESHOLD - PRECISION)  # Use full threshold range
            b_x_min = a_x_max + gap
            b_x_max = random.randint(min(int(b_x_min) + 5, WORLD_SIZE//2 - 5), WORLD_SIZE//2)
            # Overlap in y dimension for horizontal adjacency
            overlap_y_min = max(a_y_min, -WORLD_SIZE//2)
            overlap_y_max = min(a_y_max, WORLD_SIZE//2)
            if overlap_y_max > overlap_y_min + 2:
                b_y_min = random.randint(int(overlap_y_min), int(overlap_y_max) - 2)
                b_y_max = random.randint(b_y_min + 5, min(int(overlap_y_max) + 5, WORLD_SIZE//2))
            else:
                b_y_min = a_y_min
                b_y_max = a_y_max + 2
        else:
            # B is above A (touching)
            gap = random.uniform(0, ADJACENT_THRESHOLD - PRECISION)  # Use full threshold range
            b_y_min = a_y_max + gap
            b_y_max = random.randint(min(int(b_y_min) + 5, WORLD_SIZE//2 - 5), WORLD_SIZE//2)
            # Overlap in x dimension for vertical adjacency
            overlap_x_min = max(a_x_min, -WORLD_SIZE//2)
            overlap_x_max = min(a_x_max, WORLD_SIZE//2)
            if overlap_x_max > overlap_x_min + 2:
                b_x_min = random.randint(int(overlap_x_min), int(overlap_x_max) - 2)
                b_x_max = random.randint(b_x_min + 5, min(int(overlap_x_max) + 5, WORLD_SIZE//2))
            else:
                b_x_min = a_x_min
                b_x_max = a_x_max + 2
        box_b = [b_x_min, b_y_min, b_x_max, b_y_max]
        
        if boxes_adjacent(box_a, box_b):
            pos_examples.append(f'adjacent({a_x_min},{a_y_min},{a_x_max},{a_y_max},{b_x_min},{b_y_min},{b_x_max},{b_y_max})')
    
    # Generate negative examples: boxes are not adjacent
    while len(neg_examples) < num_neg:
        # Generate first bounding box
        a_x_min = random.randint(-WORLD_SIZE//2, WORLD_SIZE//4)
        a_x_max = random.randint(a_x_min + 5, WORLD_SIZE//2)
        a_y_min = random.randint(-WORLD_SIZE//2, WORLD_SIZE//4)
        a_y_max = random.randint(a_y_min + 5, WORLD_SIZE//2)
        box_a = [a_x_min, a_y_min, a_x_max, a_y_max]
        
        # Generate second bounding box that's far away or overlapping
        if random.random() < 0.5:
            # B is far from A
            b_x_min = random.randint(a_x_max + int(ADJACENT_THRESHOLD) + 1, WORLD_SIZE - 10)
            b_x_max = random.randint(min(b_x_min + 5, WORLD_SIZE - 1), WORLD_SIZE)
            b_y_min = random.randint(-WORLD_SIZE//2, WORLD_SIZE//2 - 10)
            b_y_max = random.randint(b_y_min + 5, WORLD_SIZE//2)
        else:
            # B overlaps with A
            b_x_min = random.randint(a_x_min + 1, a_x_max - 1)
            b_x_max = random.randint(b_x_min + 5, a_x_max + 10)
            b_y_min = random.randint(a_y_min + 1, a_y_max - 1)
            b_y_max = random.randint(b_y_min + 5, a_y_max + 10)
        box_b = [b_x_min, b_y_min, b_x_max, b_y_max]
        
        if not boxes_adjacent(box_a, box_b):
            neg_examples.append(f'adjacent({a_x_min},{a_y_min},{a_x_max},{a_y_max},{b_x_min},{b_y_min},{b_x_max},{b_y_max})')
    
    return pos_examples, neg_examples, {}


# Problem 8: aligned
def generate_aligned(num_pos=50, num_neg=50):
    """Generate aligned problem: Three points are collinear"""
    pos_examples = []
    neg_examples = []
    
    def points_collinear(p1, p2, p3):
        """Check if three points are collinear using cross product"""
        cross = (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0])
        return abs(cross) < PRECISION
    
    # Generate positive examples: three points are collinear
    while len(pos_examples) < num_pos:
        # Generate a line first (two points)
        p1 = gen_point_2d()
        p2 = gen_point_2d()
        # Ensure p1 and p2 are different (not same point)
        if abs(p1[0] - p2[0]) < 1 and abs(p1[1] - p2[1]) < 1:
            continue
        # Generate third point on the line
        t = random.uniform(-1.5, 2.5)  # Can be beyond the segment
        p3_x = p1[0] + t * (p2[0] - p1[0])
        p3_y = p1[1] + t * (p2[1] - p1[1])
        # Keep within bounds
        p3_x = max(-WORLD_SIZE, min(WORLD_SIZE, p3_x))
        p3_y = max(-WORLD_SIZE, min(WORLD_SIZE, p3_y))
        p3 = [p3_x, p3_y]
        
        if points_collinear(p1, p2, p3):
            pos_examples.append(f'aligned({point_to_string(p1)},{point_to_string(p2)},{point_to_string(p3)})')
    
    # Generate negative examples: three points are not collinear
    while len(neg_examples) < num_neg:
        # Generate three random points (likely not collinear)
        p1 = gen_point_2d()
        p2 = gen_point_2d()
        p3 = gen_point_2d()
        
        if not points_collinear(p1, p2, p3):
            neg_examples.append(f'aligned({point_to_string(p1)},{point_to_string(p2)},{point_to_string(p3)})')
    
    return pos_examples, neg_examples, {}


# Problem 9: surrounds
def generate_surrounds(num_pos=50, num_neg=50):
    """Generate surrounds problem: Bounding box A surrounds bounding box B"""
    pos_examples = []
    neg_examples = []
    
    def box_surrounds(outer, inner):
        """Check if outer box surrounds inner box"""
        ox_min, oy_min, ox_max, oy_max = outer
        ix_min, iy_min, ix_max, iy_max = inner
        return (ox_min < ix_min and ix_max < ox_max and
                oy_min < iy_min and iy_max < oy_max)
    
    # Generate positive examples: A surrounds B
    while len(pos_examples) < num_pos:
        # Generate outer box A (ensure it's large enough)
        a_x_min = random.randint(-WORLD_SIZE//2, WORLD_SIZE//4)
        a_x_max = random.randint(a_x_min + 20, WORLD_SIZE//2)
        a_y_min = random.randint(-WORLD_SIZE//2, WORLD_SIZE//4)
        a_y_max = random.randint(a_y_min + 20, WORLD_SIZE//2)
        box_a = [a_x_min, a_y_min, a_x_max, a_y_max]
        
        # Generate inner box B inside A - ensure valid bounds
        if a_x_max - a_x_min < 14 or a_y_max - a_y_min < 14:
            continue  # Skip if outer box too small
        
        b_x_min = random.randint(a_x_min + 2, a_x_max - 12)
        b_x_max = random.randint(b_x_min + 5, a_x_max - 2)
        b_y_min = random.randint(a_y_min + 2, a_y_max - 12)
        b_y_max = random.randint(b_y_min + 5, a_y_max - 2)
        box_b = [b_x_min, b_y_min, b_x_max, b_y_max]
        
        if box_surrounds(box_a, box_b):
            pos_examples.append(f'surrounds({a_x_min},{a_y_min},{a_x_max},{a_y_max},{b_x_min},{b_y_min},{b_x_max},{b_y_max})')
    
    # Generate negative examples: A does not surround B
    while len(neg_examples) < num_neg:
        # Generate box A
        a_x_min = random.randint(-WORLD_SIZE//2, WORLD_SIZE//4)
        a_x_max = random.randint(a_x_min + 10, WORLD_SIZE//2)
        a_y_min = random.randint(-WORLD_SIZE//2, WORLD_SIZE//4)
        a_y_max = random.randint(a_y_min + 10, WORLD_SIZE//2)
        box_a = [a_x_min, a_y_min, a_x_max, a_y_max]
        
        # Generate box B that's not fully inside A - simple approach
        if random.random() < 0.5:
            # B extends outside A horizontally
            b_x_min = random.randint(max(-WORLD_SIZE//2, a_x_min - 5), min(a_x_max - 2, WORLD_SIZE//2 - 15))
            b_x_max = random.randint(max(b_x_min + 5, a_x_max + 1), WORLD_SIZE//2)
            b_y_min = random.randint(a_y_min + 2, max(a_y_min + 2, a_y_max - 5))
            b_y_max = random.randint(max(b_y_min + 5, b_y_min + 5), max(b_y_min + 5, a_y_max - 2))
        else:
            # B extends outside A vertically
            b_x_min = random.randint(a_x_min + 2, max(a_x_min + 2, a_x_max - 5))
            b_x_max = random.randint(max(b_x_min + 5, b_x_min + 5), max(b_x_min + 5, a_x_max - 2))
            b_y_min = random.randint(max(-WORLD_SIZE//2, a_y_min - 5), min(a_y_max - 2, WORLD_SIZE//2 - 15))
            b_y_max = random.randint(max(b_y_min + 5, a_y_max + 1), WORLD_SIZE//2)
        
        box_b = [b_x_min, b_y_min, b_x_max, b_y_max]
        
        if not box_surrounds(box_a, box_b):
            neg_examples.append(f'surrounds({a_x_min},{a_y_min},{a_x_max},{a_y_max},{b_x_min},{b_y_min},{b_x_max},{b_y_max})')
    
    return pos_examples, neg_examples, {}


# Problem 10: near_corner
def generate_near_corner(num_pos=50, num_neg=50):
    """Generate near_corner problem: Point A is near a corner of bounding box B"""
    pos_examples = []
    neg_examples = []
    
    def distance_to_corner(point, box):
        """Calculate minimum distance from point to any corner of box"""
        x, y = point
        x_min, y_min, x_max, y_max = box
        corners = [
            (x_min, y_min), (x_max, y_min),
            (x_min, y_max), (x_max, y_max)
        ]
        min_dist = float('inf')
        for corner in corners:
            dist = distance_2d(point, list(corner))
            min_dist = min(min_dist, dist)
        return min_dist
    
    # Generate positive examples: point is near a corner
    while len(pos_examples) < num_pos:
        # Generate bounding box
        b_x_min = random.randint(-WORLD_SIZE//2, WORLD_SIZE//4)
        b_x_max = random.randint(b_x_min + 10, WORLD_SIZE//2)
        b_y_min = random.randint(-WORLD_SIZE//2, WORLD_SIZE//4)
        b_y_max = random.randint(b_y_min + 10, WORLD_SIZE//2)
        box = [b_x_min, b_y_min, b_x_max, b_y_max]
        
        # Generate point near a corner
        corner_idx = random.randint(0, 3)
        corner_range = NEAR_CORNER_THRESHOLD  # Use full threshold range
        if corner_idx == 0:  # Bottom-left
            a_x = random.uniform(b_x_min - corner_range, b_x_min + corner_range)
            a_y = random.uniform(b_y_min - corner_range, b_y_min + corner_range)
        elif corner_idx == 1:  # Bottom-right
            a_x = random.uniform(b_x_max - corner_range, b_x_max + corner_range)
            a_y = random.uniform(b_y_min - corner_range, b_y_min + corner_range)
        elif corner_idx == 2:  # Top-left
            a_x = random.uniform(b_x_min - corner_range, b_x_min + corner_range)
            a_y = random.uniform(b_y_max - corner_range, b_y_max + corner_range)
        else:  # Top-right
            a_x = random.uniform(b_x_max - corner_range, b_x_max + corner_range)
            a_y = random.uniform(b_y_max - corner_range, b_y_max + corner_range)
        
        a = [a_x, a_y]
        dist = distance_to_corner(a, box)
        if dist <= NEAR_CORNER_THRESHOLD - PRECISION:
            pos_examples.append(f'near_corner({point_to_string(a)},{b_x_min},{b_y_min},{b_x_max},{b_y_max})')
    
    # Generate negative examples: point is far from corners
    while len(neg_examples) < num_neg:
        # Generate bounding box
        b_x_min = random.randint(-WORLD_SIZE//2, WORLD_SIZE//4)
        b_x_max = random.randint(b_x_min + 10, WORLD_SIZE//2)
        b_y_min = random.randint(-WORLD_SIZE//2, WORLD_SIZE//4)
        b_y_max = random.randint(b_y_min + 10, WORLD_SIZE//2)
        box = [b_x_min, b_y_min, b_x_max, b_y_max]
        
        # Generate point far from corners (near center or middle of edges)
        if random.random() < 0.5:
            # Near center
            a_x = random.uniform(b_x_min + (b_x_max - b_x_min) * 0.3, b_x_min + (b_x_max - b_x_min) * 0.7)
            a_y = random.uniform(b_y_min + (b_y_max - b_y_min) * 0.3, b_y_min + (b_y_max - b_y_min) * 0.7)
        else:
            # Near middle of an edge
            a_x = random.uniform(b_x_min + (b_x_max - b_x_min) * 0.3, b_x_min + (b_x_max - b_x_min) * 0.7)
            a_y = random.uniform(b_y_min - NEAR_CORNER_THRESHOLD - 5, b_y_min - NEAR_CORNER_THRESHOLD - 1)
        
        a = [a_x, a_y]
        dist = distance_to_corner(a, box)
        if dist > NEAR_CORNER_THRESHOLD + PRECISION:
            neg_examples.append(f'near_corner({point_to_string(a)},{b_x_min},{b_y_min},{b_x_max},{b_y_max})')
    
    return pos_examples, neg_examples, {'threshold': NEAR_CORNER_THRESHOLD}


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
        f.write("% Background knowledge for geometry2 problems\n")
        f.write("% This file is a template - the learner will add feature facts dynamically\n\n")
        
        if problem_type == 'left_of':
            f.write("% left_of(A, B): A is to the left of B\n")
            f.write("% Mode: left_of(+point, +point)\n")
        elif problem_type == 'closer_than':
            f.write("% closer_than(A, B, C): A is closer to B than C is to B\n")
            f.write("% Mode: closer_than(+point, +point, +point)\n")
        elif problem_type == 'touching':
            f.write("% touching(A, B): A and B are touching (within threshold distance)\n")
            f.write("% Mode: touching(+point, +point)\n")
        elif problem_type == 'inside':
            f.write("% inside(A, B): A is inside B's bounding box\n")
            f.write("% Mode: inside(+point, +bbox)\n")
        elif problem_type == 'overlapping':
            f.write("% overlapping(A, B): Two bounding boxes overlap\n")
            f.write("% Mode: overlapping(+bbox, +bbox)\n")
        elif problem_type == 'between':
            f.write("% between(A, B, C): A is between B and C (collinear and between)\n")
            f.write("% Mode: between(+point, +point, +point)\n")
        elif problem_type == 'adjacent':
            f.write("% adjacent(A, B): Two bounding boxes are adjacent (touching at edges)\n")
            f.write("% Mode: adjacent(+bbox, +bbox)\n")
        elif problem_type == 'aligned':
            f.write("% aligned(A, B, C): Three points are collinear\n")
            f.write("% Mode: aligned(+point, +point, +point)\n")
        elif problem_type == 'surrounds':
            f.write("% surrounds(A, B): Bounding box A surrounds bounding box B\n")
            f.write("% Mode: surrounds(+bbox, +bbox)\n")
        elif problem_type == 'near_corner':
            f.write("% near_corner(A, B): Point A is near a corner of bounding box B\n")
            f.write("% Mode: near_corner(+point, +bbox)\n")


if __name__ == "__main__":
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    os.makedirs(data_dir, exist_ok=True)
    
    print("Generating Geometry2 datasets...")
    
    # Generate left_of
    print("Generating left_of problem...")
    pos, neg, params = generate_left_of()
    write_examples_file(os.path.join(data_dir, 'left_of_examples.pl'), pos, neg)
    write_bk_file(os.path.join(data_dir, 'left_of_BK.pl'), 'left_of')
    print(f"  Generated {len(pos)} positive and {len(neg)} negative examples")
    
    # Generate closer_than
    print("Generating closer_than problem...")
    pos, neg, params = generate_closer_than()
    write_examples_file(os.path.join(data_dir, 'closer_than_examples.pl'), pos, neg)
    write_bk_file(os.path.join(data_dir, 'closer_than_BK.pl'), 'closer_than')
    print(f"  Generated {len(pos)} positive and {len(neg)} negative examples")
    
    # Generate touching
    print("Generating touching problem...")
    pos, neg, params = generate_touching()
    write_examples_file(os.path.join(data_dir, 'touching_examples.pl'), pos, neg)
    write_bk_file(os.path.join(data_dir, 'touching_BK.pl'), 'touching')
    print(f"  Generated {len(pos)} positive and {len(neg)} negative examples")
    
    # Generate inside
    print("Generating inside problem...")
    pos, neg, params = generate_inside()
    write_examples_file(os.path.join(data_dir, 'inside_examples.pl'), pos, neg)
    write_bk_file(os.path.join(data_dir, 'inside_BK.pl'), 'inside')
    print(f"  Generated {len(pos)} positive and {len(neg)} negative examples")
    
    # Generate overlapping
    print("Generating overlapping problem...")
    pos, neg, params = generate_overlapping()
    write_examples_file(os.path.join(data_dir, 'overlapping_examples.pl'), pos, neg)
    write_bk_file(os.path.join(data_dir, 'overlapping_BK.pl'), 'overlapping')
    print(f"  Generated {len(pos)} positive and {len(neg)} negative examples")
    
    # Generate between
    print("Generating between problem...")
    pos, neg, params = generate_between()
    write_examples_file(os.path.join(data_dir, 'between_examples.pl'), pos, neg)
    write_bk_file(os.path.join(data_dir, 'between_BK.pl'), 'between')
    print(f"  Generated {len(pos)} positive and {len(neg)} negative examples")
    
    # Generate adjacent
    print("Generating adjacent problem...")
    pos, neg, params = generate_adjacent()
    write_examples_file(os.path.join(data_dir, 'adjacent_examples.pl'), pos, neg)
    write_bk_file(os.path.join(data_dir, 'adjacent_BK.pl'), 'adjacent')
    print(f"  Generated {len(pos)} positive and {len(neg)} negative examples")
    
    # Generate aligned
    print("Generating aligned problem...")
    pos, neg, params = generate_aligned()
    write_examples_file(os.path.join(data_dir, 'aligned_examples.pl'), pos, neg)
    write_bk_file(os.path.join(data_dir, 'aligned_BK.pl'), 'aligned')
    print(f"  Generated {len(pos)} positive and {len(neg)} negative examples")
    
    # Generate surrounds
    print("Generating surrounds problem...")
    pos, neg, params = generate_surrounds()
    write_examples_file(os.path.join(data_dir, 'surrounds_examples.pl'), pos, neg)
    write_bk_file(os.path.join(data_dir, 'surrounds_BK.pl'), 'surrounds')
    print(f"  Generated {len(pos)} positive and {len(neg)} negative examples")
    
    # Generate near_corner
    print("Generating near_corner problem...")
    pos, neg, params = generate_near_corner()
    write_examples_file(os.path.join(data_dir, 'near_corner_examples.pl'), pos, neg)
    write_bk_file(os.path.join(data_dir, 'near_corner_BK.pl'), 'near_corner')
    print(f"  Generated {len(pos)} positive and {len(neg)} negative examples")
    
    print("\nAll Geometry2 datasets generated successfully!")
