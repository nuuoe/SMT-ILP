# NumSynth-Compatible Datasets

This directory contains all datasets converted to NumSynth format for fair comparison between PyGol+Z3 and NumSynth.

## Structure

Each problem is in its own folder with three files:
- `exs.pl` - Examples in NumSynth format (`pos(...)` and `neg(...)`)
- `bk.pl` - Background knowledge with NumSynth-compatible arithmetic operations
- `numsynth-bias.pl` - Bias file for NumSynth

## Datasets

### Geometry0 (2 problems)
- `interval/` - 1D interval problem
- `halfplane/` - 2D halfplane problem

### Geometry1 (4 problems)
- `halfplane3d/` - 3D halfplane
- `conjunction/` - Conjunction of constraints
- `multihalfplane/` - Multiple halfplanes
- `interval3d/` - 3D interval

### Geometry2 (10 problems)
- `left_of/` - Relational: left_of(A, B)
- `inside/` - Relational: inside(A, B)
- `touching/` - Relational: touching(A, B)
- `overlapping/` - Relational: overlapping(A, B)
- `between/` - Relational: between(A, B, C)
- `aligned/` - Relational: aligned(A, B, C)
- `closer_than/` - Relational: closer_than(A, B, C)
- `near_corner/` - Relational: near_corner(A, B)
- `adjacent/` - Relational: adjacent(A, B)
- `surrounds/` - Relational: surrounds(A, B)

### Geometry3 (14 problems)
- `in_circle/` - Nonlinear: circle region
- `in_ellipse/` - Nonlinear: ellipse region
- `hyperbola_side/` - Nonlinear: hyperbola
- `xy_less_than/` - Nonlinear: xy < threshold
- `quad_strip/` - Nonlinear: quadratic strip
- `union_halfplanes/` - Disjunctive: union of halfplanes
- `circle_or_box/` - Disjunctive: circle OR box
- `piecewise/` - Disjunctive: piecewise function
- `fallback_region/` - Disjunctive: fallback region
- `donut/` - Non-convex: donut shape
- `lshape/` - Non-convex: L-shape
- `above_parabola/` - Non-convex: above parabola
- `sinusoidal/` - Non-convex: sinusoidal region
- `crescent/` - Non-convex: crescent shape

### IP (5 problems)
- `ip1_active/` - Single-hop influence propagation
- `ip2_active/` - 2-hop chain (requires predicate invention)
- `ip3_active/` - 3-hop chain (requires predicate invention)
- `ip3_threshold/` - 3-hop + threshold (requires PI + Z3)
- `ip4_high_score/` - 3-hop + aggregate score (requires PI + Z3)

## Usage with NumSynth

To run NumSynth on a problem:

```bash
cd /path/to/numsynth
python popper.py /path/to/numsynth_datasets/geometry0/interval --numerical-reasoning
```

## Conversion Notes

- All examples are converted from the original format to NumSynth's `pos(...)` / `neg(...)` format
- Background knowledge is adapted to include NumSynth's arithmetic operations (`add`, `mult`, `leq`, `geq`)
- Bias files are created with appropriate settings for each problem type
- **Important**: These are the exact same examples used by PyGol+Z3, ensuring fair comparison

## Total Problems

- **35 problems** across all datasets
- All problems have examples, BK, and bias files ready for NumSynth

