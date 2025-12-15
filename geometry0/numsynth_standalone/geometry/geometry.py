"""
Standalone GeometryProblem base class - minimal version without numsynth dependencies
"""
import os
import random

WORLD_SIZE = 100

def gen_point(min_val=None, max_val=None, dimension=1):
    """Generate a random point in the specified dimension"""
    if dimension and not min_val and not max_val:
        min_val = [-WORLD_SIZE for _ in range(dimension)]
        max_val = [WORLD_SIZE for _ in range(dimension)]
    return [random.randint(min_v, max_v) for min_v, max_v in zip(*[min_val, max_val])]


class GeometryProblem:
    """
    Minimal base class for geometry problems.
    Simplified version that only provides what's needed for data generation.
    """
    
    def __init__(self, name, gen_pos, gen_neg, sub_dir):
        self.name = name
        self.gen_pos = gen_pos
        self.gen_neg = gen_neg
        self.sub_dir = sub_dir
    
    def bk_file(self):
        """Return path to BK file - simplified version"""
        # Return None as the learner creates its own BK
        return None

