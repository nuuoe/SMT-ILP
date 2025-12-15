# SMT-ILP: Hybrid PyGol+Z3 Learning Experiments

This repository contains experiments demonstrating the integration of **PyGol** (Inductive Logic Programming) with **Z3** (Satisfiability Modulo Theories) for learning hybrid symbolic-numeric rules. The benchmark suite consists of 35 problems across 5 datasets, testing everything from basic linear constraints to nonlinear geometric regions and multi-hop relational reasoning.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Dataset-Specific Instructions](#dataset-specific-instructions)
- [Running All Experiments](#running-all-experiments)
- [Understanding Output](#understanding-output)
- [Troubleshooting](#troubleshooting)

---

## Prerequisites

**Required Software:**
- Python 3.7+
- SWI-Prolog 9.2.0+ ([download](https://www.swi-prolog.org)) - ensure `swipl` is in PATH
- Clingo 5.5.0+ ([download](https://potassco.org/clingo/)) - ensure `clingo` is in PATH

**Python Dependencies:**
```bash
pip install numpy pandas scikit-learn z3-solver
```

If using conda (recommended):
```bash
conda activate pygol-z3-zendo
```

---

## Installation

1. Ensure PyGol is installed with `pygol.so` available (pre-compiled for Mac M1, compile from source for other systems). If PyGol is in a parent directory, ensure it's in your Python path.

2. Verify installation:
   ```bash
   python -c "from pygol import *; print('PyGol OK')"
   python -c "import z3; print('Z3 version:', z3.get_version_string())"
   ```

---

## Quick Start

### Run a Single Quick Test

Test Geometry0 (Interval problem) - fastest test (~1-2 seconds):

```bash
cd geometry0
python test_geometry0_quick.py
```

### Generate Data

Generate data files before running experiments (uses fixed random seeds for reproducibility):

```bash
for dir in geometry0 geometry1 geometry2 geometry3 ip; do
    cd $dir && python generate_${dir}.py && cd ..
done
```

---

## Dataset-Specific Instructions

### Geometry0: Basic Linear Constraints

**Location**: `geometry0/`

**Overview**: Basic 2D geometry problems testing fundamental linear arithmetic constraint learning. These are the original NumSynth problems.

**Problems**: 2 tasks
1. **Interval** - Learning rules like `interval(X) :- lower < X < upper`
   - Univariate interval constraint where positive examples fall within a specific range
2. **Halfplane** - Learning rules like `halfplane(X, Y) :- coeff1*X + coeff2*Y <= threshold`
   - 2D linear halfplane constraint where positive examples satisfy a linear inequality

**Files**:
- `iterative_pygol_z3_learner_geometry0.py` - Main geometry0-specific learner implementation
- `generate_geometry0.py` - Data generation script using standalone numsynth problem classes
- `load_geometry0_data.py` - Data loading functions for geometry0 problems
- `test_geometry0_all.py` - Full test script for all geometry0 problems
- `test_geometry0_quick.py` - Quick test script for rapid verification (tests interval)
- `numsynth_standalone/` - Standalone copy of numsynth geometry problem classes (no external dependency)

**Data Files** (in `data/` folder):
- `interval_examples.pl`
- `halfplane_examples.pl`

The learner creates its own BK files with arithmetic operations dynamically.

**Quick Test** (single problem):
```bash
cd geometry0
python test_geometry0_quick.py
```

**Full Test** (both problems):
```bash
cd geometry0
python test_geometry0_all.py
```

**Expected Runtime**: 1-3 seconds per problem

**Technical Notes**:
- Arithmetic learning with coefficient bounds (-100, 100)
- Range rules for intervals (lower < X < upper), linear rules for halfplanes
- Rules output in BK format using `leq`, `geq`, `mult`, `add` predicates
- Fixed random seeds for reproducibility
- Includes standalone numsynth geometry classes (no external dependency)

---

### Geometry1: 3D and Conjunctive Constraints

**Location**: `geometry1/`

**Overview**: 3D linear arithmetic and conjunctive constraint learning. More challenging than Geometry0.

**Problems**: 4 tasks
1. **3D Halfplane** - Learning rules like `halfplane3d(X, Y, Z) :- a*X + b*Y + c*Z <= d`
   - Extends halfplane learning to 3D space, learning a linear constraint across three dimensions
2. **Conjunction** - Learning rules combining halfplane and interval: `(x + y <= threshold) AND (z_min < z < z_max)`
   - Learn conjunctive constraints combining a 2D halfplane with a 1D interval
3. **Multiple Halfplanes** - Learning rules with multiple constraints: `(a1*x + b1*y <= c1) AND (a2*x + b2*y <= c2)`
   - Learn multiple simultaneous halfplane constraints defining a polygonal region
4. **3D Interval** - Learning rules with 3D intervals: `(x_min < x < x_max) AND (y_min < y < y_max) AND (z_min < z < z_max)`
   - Learn a 3D axis-aligned bounding box (rectangular region)

**Files**:
- `iterative_pygol_z3_learner_geometry1.py` - Main geometry1-specific learner implementation
- `generate_geometry1.py` - Data generation script for all geometry1 problems
- `load_geometry1_data.py` - Data loading functions for geometry1 problems
- `test_geometry1_all.py` - Full test script for all geometry1 problems
- `test_geometry1_quick.py` - Quick test script for rapid verification

**Data Files** (in `data/` folder):
- `halfplane3d_examples.pl` / `halfplane3d_BK.pl`
- `conjunction_examples.pl` / `conjunction_BK.pl`
- `multihalfplane_examples.pl` / `multihalfplane_BK.pl`
- `interval3d_examples.pl` / `interval3d_BK.pl`

The learner creates its own BK files with arithmetic operations dynamically, so the BK files in `data/` are mainly for reference.

**Quick Test** (3D Halfplane only):
```bash
cd geometry1
python test_geometry1_quick.py
```

**Full Test** (all 4 problems):
```bash
cd geometry1
python test_geometry1_all.py
```

**Expected Runtime**: 15-45 seconds per problem

**Technical Notes**:
- Arithmetic learning with bounds (-100, 100)
- More challenging than Geometry0; may need longer timeouts
- Iterative refinement improves rule quality across multiple iterations
- Supports 3-variable combinations for 3D linear relationships

---

### Geometry2: Relational Spatial Constraints

**Location**: `geometry2/`

**Overview**: Relational spatial reasoning with multi-argument predicates. Tests constraints requiring comparisons between multiple head arguments.

**Problems**: 10 tasks
- **left_of(A, B)** - A is to the left of B (x-coordinate comparison)
- **closer_than(A, B, C)** - A is closer to B than C is to B
- **touching(A, B)** - A and B are touching (within threshold distance)
- **inside(A, B)** - A is inside B (A's bounding box is within B's bounding box)
- **overlapping(A, B)** - Two bounding boxes overlap
- **between(A, B, C)** - A is between B and C (collinear and between)
- **adjacent(A, B)** - Two bounding boxes are adjacent (touching at edges)
- **aligned(A, B, C)** - Three points are collinear
- **surrounds(A, B)** - Bounding box A surrounds bounding box B
- **near_corner(A, B)** - Point A is near a corner of bounding box B

**Files**:
- `iterative_pygol_z3_learner_geometry2.py` - Main geometry2-specific learner implementation
- `generate_geometry2.py` - Data generation script for all geometry2 problems
- `load_geometry2_data.py` - Data loading functions for geometry2 problems
- `test_geometry2_all.py` - Full test script for all geometry2 problems
- `test_geometry2_quick.py` - Quick test script for rapid verification (tests left_of)

**Data Files** (in `data/` folder):
- Example files and BK files for all 10 problems (e.g., `left_of_examples.pl`, `left_of_BK.pl`, etc.)

**Quick Test** (single problem):
```bash
cd geometry2
python test_geometry2_quick.py
```

**Full Test** (all 10 problems):
```bash
cd geometry2
python test_geometry2_all.py
```

**Expected Runtime**: 20-60 seconds per problem

**Technical Notes**:
- Requires relational reasoning with multi-argument predicates
- Rules involve variable-variable comparisons (e.g., `X_1 < X_2`)
- PyGol discovers relational patterns; Z3 refines thresholds
- Rich spatial background knowledge (distance, angle, bounding boxes)

---

### Geometry3: Nonlinear and Disjunctive Constraints

**Location**: `geometry3/`

**Overview**: Nonlinear and disjunctive geometric constraints testing hybrid SMT-ILP expressiveness. These tasks cannot be represented in classical ILP or linear-only numerical ILP systems.

**Problems**: 14 tasks organized into three categories

**Category 1: Nonlinear Regions (5 problems)**
1. **in_circle**: x² + y² < r²
2. **in_ellipse**: (x-h)²/a² + (y-k)²/b² < 1
3. **hyperbola_side**: x² - y² > k
4. **xy_less_than**: x·y < c
5. **quad_strip**: a < x² < b

**Category 2: Disjunctive (OR) Regions (4 problems)**
6. **union_halfplanes**: (ax + by ≤ c) OR (dx + ey ≥ f)
7. **circle_or_box**: (x² + y² < r²) OR (|x| < s AND |y| < s)
8. **piecewise**: (x < 0 AND y > 5) OR (x > 10)
9. **fallback_region**: in_circle OR below_line

**Category 3: Non-convex / Piecewise / Hybrid Regions (5 problems)**
10. **donut**: r₁² < x² + y² < r₂²
11. **lshape**: Union of two rectangles
12. **above_parabola**: y > ax² + bx + c
13. **sinusoidal**: y > sin(scale·x) + offset
14. **crescent**: (x² + y² < r₁²) AND ((x-h)² + (y-k)² > r₂²)

**Files**:
- `iterative_pygol_z3_learner_geometry3.py` - Main geometry3-specific learner implementation
- `generate_geometry3.py` - Data generation script for all 14 geometry3 problems
- `load_geometry3_data.py` - Data loading functions for geometry3 problems
- `test_geometry3_all.py` - Full test script for all 14 geometry3 problems
- `test_geometry3_quick.py` - Quick test script for rapid verification

**Data Files** (in `data/` folder):
- Example files and BK files for all 14 problems (e.g., `in_circle_examples.pl`, `in_circle_BK.pl`, etc.)

**What Geometry3 Tests**:
- Nonlinear arithmetic: quadratic, multiplicative, trigonometric constraints
- Disjunctive reasoning: multiple alternative hypotheses (implicit OR)
- Non-convex regions: complex shapes requiring predicate composition
- Hybrid reasoning: PyGol structure learning + Z3 numeric optimization

**Quick Test** (single problem):
```bash
cd geometry3
python test_geometry3_quick.py
```

**Full Test** (all 14 problems):
```bash
cd geometry3
python test_geometry3_all.py
```

**Expected Runtime**: 30-120 seconds per problem

**Technical Notes**:
- Tests Z3's nonlinear arithmetic (quadratic, trigonometric, multiplicative)
- PyGol proposes region decompositions; Z3 instantiates nonlinear parameters
- SMT reasoning is essential: ILP alone cannot express these numerical relations

---

### IP: Multi-Hop Relational Reasoning with Predicate Invention

**Location**: `ip/`

**Overview**: Relational reasoning benchmark demonstrating PyGol's Predicate Invention for multi-hop propagation chains. Unlike Geometry datasets, the primary challenge is discovering multi-hop relational patterns (influence computations involve nonlinear arithmetic but are secondary).

**Dataset Description**:
- **50 objects** with score and 2D position
- **Influence function**: `(score_A × score_B) / dist(A,B)²`
- **Propagates relation**: touching(A,B) AND influence(A,B) > threshold
- **5 tasks** with progressive difficulty
- **Random seed**: 1234 (for reproducibility)

**Problems**: 5 tasks

1. **ip1_active (Baseline)**
   ```prolog
   ip1_active(A) :- propagates(A,B), propagates(B,A).
   ```
   - 2 literals (mutual propagation)
   - Works without PI: 50%
   - Purpose: Validates baseline system

2. **ip2_active (PI Required)**
   ```prolog
   ip2_active(A) :- propagates(A,B), propagates(B,C), propagates(C,A).
   ```
   - 3 literals (triangle pattern)
   - Without PI: N/A (cannot learn)
   - With PI: 54-60%
   - Purpose: Demonstrates PI enables multi-hop reasoning

3. **ip3_active (PI Required)**
   - Similar to ip2_active but with longer chains
   - Requires PI for multi-hop structure

4. **ip3_threshold (PI + Numerical)**
   ```prolog
   ip3_threshold(A) :- propagates(A,B), propagates(B,C), propagates(C,A),
                        influence_constraint(A).
   ```
   - 3 literals + numerical constraint
   - Requires PI for chain structure
   - Numerical features available for Z3

5. **ip4_high_score (PI + Aggregate)**
   ```prolog
   ip4_high_score(A) :- propagates(A,B), propagates(B,C), propagates(C,D),
                         aggregate_score(A,B,C,D) > threshold.
   ```
   - 4 literals + aggregate constraint
   - Requires PI for longer chain
   - Aggregate features for Z3 optimization

**Files**:
- `generate_ip.py` - Data generation
- `load_ip_data.py` - Hybrid data loader (relational + numerical)
- `iterative_pygol_z3_learner_ip_main.py` - Main learner with Z3 refinement
- `test_ip_all.py` - Complete benchmark

**Data Files** (in `data/` folder):
- `ip1_active_examples.pl`, `ip2_active_examples.pl`, `ip3_active_examples.pl`
- `ip3_threshold_examples.pl`, `ip4_high_score_examples.pl`
- `objects_BK.pl` - Background knowledge with propagates facts

**Key Features**:
- Small relational BK (just propagates facts)
- Numerical features in DataFrame
- Two-stage learning (PyGol -> Z3 refinement)
- Custom relational prediction engine

**Full Test** (all 5 tasks, 3 configurations each):
```bash
cd ip
python test_ip_all.py
```

**Expected Runtime**: 80-160 seconds per task

**Configurations Tested**:
1. **No PI**: Predicate invention disabled (2 literals)
2. **PI Only**: Predicate invention enabled, Z3 disabled (4 literals)
3. **PI + Z3**: Full system with both PI and Z3 (4 literals)

**Technical Notes**:
- Tasks 2-5 require predicate invention for multi-hop reasoning
- Patterns cannot be represented at bounded clause depth without PI
- PyGol invents auxiliary predicates for intermediate abstractions
- Z3 optimizes numerical thresholds for influence-based predicates
- Geometry stresses SMT's numerical expressivity; IP stresses ILP's structural expressivity

---

## Running All Experiments

### Run Complete Benchmark Suite

To run all experiments across all datasets with multiple trials:

```bash
python run_all_pygol_z3.py
```

**Configuration**:
- 5 trials per problem (default, modifiable in script)
- Automatic random seed assignment for reproducibility
- Results saved to `Results/` directory

**Expected Runtime**: 
- Geometry0: ~10s | Geometry1: ~5min | Geometry2: ~10-15min | Geometry3: ~30-60min | IP: ~15-20min
- **Total**: ~1-2 hours for complete benchmark

### Output Files

Results are saved in `Results/` directory:
- `pygol_z3_results_geometry0.md`
- `pygol_z3_results_geometry1.md`
- `pygol_z3_results_geometry2.md`
- `pygol_z3_results_geometry3.md`
- `pygol_z3_results_ip.md`
- `pygol_z3_results.md` (aggregated summary)

Each results file contains test accuracy (mean ± std), training accuracy, learning time, number of rules, and per-problem breakdowns.

---

## Understanding Output

Test scripts output problem name, iteration progress, and final metrics:
- **Test/Train Accuracy**: Performance on test (70/30 split) and training sets
- **Time**: Total learning time in seconds
- **Rules Learned**: Number of rules in final hypothesis

Rules are displayed in Prolog format:
```prolog
target(A, B) :- geq(A, 10.5), leq(B, 20.3).
```
Where `geq`/`leq` are comparison predicates and numeric values are thresholds learned by Z3.

---

## Troubleshooting

**ModuleNotFoundError: No module named 'PyGol'**
- Ensure PyGol is installed and accessible. If PyGol is in a parent directory, add it to Python path: `export PYTHONPATH=/path/to/PyGol:$PYTHONPATH`
- Test scripts should handle paths automatically if PyGol is properly installed

**SWI-Prolog/Clingo not found**
- Install from [SWI-Prolog](https://www.swi-prolog.org) and [Clingo](https://potassco.org/clingo/)
- Verify: `swipl --version` and `clingo --version`
- Ensure both are in PATH

**Z3 Import Error**
- `pip install z3-solver`

**Data files not found**
- Generate data first: `cd <dataset_directory> && python generate_<dataset>.py`

**Timeout Errors**
- Increase `pygol_timeout` in test scripts (default varies by dataset)

**Memory Issues (Geometry3/IP)**
- Reduce `max_iterations` or `max_literals`
- Process problems individually

**Platform Notes**:
- Mac M1/M2: Pre-compiled `pygol.so` included; recompile if needed
- Linux: May need to compile `pygol.so` from source (requires SWI-Prolog dev headers)
- Windows: Use WSL or conda environment

---

## File Structure

```
SMT-ILP/
├── README.md
├── run_all_pygol_z3.py          # Run all experiments
├── geometry0/                    # 2 problems
│   ├── generate_geometry0.py
│   ├── test_geometry0_all.py
│   ├── test_geometry0_quick.py
│   └── data/
├── geometry1/                    # 4 problems
│   ├── generate_geometry1.py
│   ├── test_geometry1_all.py
│   └── data/
├── geometry2/                    # 10 problems
│   ├── generate_geometry2.py
│   ├── test_geometry2_all.py
│   └── data/
├── geometry3/                    # 14 problems
│   ├── generate_geometry3.py
│   ├── test_geometry3_all.py
│   └── data/
├── ip/                           # 5 problems
│   ├── generate_ip.py
│   ├── test_ip_all.py
│   └── data/
└── Results/                      # Experiment results
    └── pygol_z3_results_*.md
```

---

## Additional Resources

- **PyGol Documentation**: See main repository README
- **Z3 Documentation**: https://github.com/Z3Prover/z3

---

## Summary

This benchmark suite contains **35 problems** across **5 datasets** evaluating SMT-ILP integration for hybrid symbolic-numeric rule learning:

- **Geometry0-3**: Progressive complexity (2D → 3D → relational → nonlinear)
- **IP**: Multi-hop relational reasoning requiring Predicate Invention
- **Coverage**: Linear/nonlinear arithmetic, disjunctive regions, relational spatial reasoning, multi-hop propagation

Together, these datasets demonstrate the full capabilities of PyGol+Z3: Z3's numerical expressivity (Geometry) and ILP's structural expressivity with Predicate Invention (IP).

