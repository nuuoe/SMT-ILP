"""
Iterative PyGol+Z3 Learner
Implements iterative refinement: PyGol → Z3 SMT → PyGol → repeat until convergence
"""

from PyGol import (
    pygol_learn, 
    bottom_clause_generation, 
    pygol_train_test_split,
    pygol_cross_validation,
    pygol_folds
)
from z3 import Solver, Real, Int, And, Or, sat, Optimize, Sum, If
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.base import BaseEstimator, ClassifierMixin
from typing import Dict, List, Any, Tuple, Optional
import re
import pandas as pd
import numpy as np
import sys
import os

# Add PyGol to path (try multiple locations for robustness)
def _find_pygol_root():
    """Find PyGol root directory by checking common locations."""
    if 'PYGOL_ROOT' in os.environ:
        pygol_root = os.environ['PYGOL_ROOT']
        if os.path.exists(pygol_root) and os.path.exists(os.path.join(pygol_root, 'pygol.so')):
            return pygol_root
    current_file = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_file)
    smt_ilp_dir = os.path.dirname(current_dir)
    pygol_root = os.path.join(os.path.dirname(smt_ilp_dir), 'PyGol')
    if os.path.exists(pygol_root) and os.path.exists(os.path.join(pygol_root, 'pygol.so')):
        return pygol_root
    for _ in range(3):
        smt_ilp_dir = os.path.dirname(smt_ilp_dir)
        pygol_root = os.path.join(smt_ilp_dir, 'PyGol')
        if os.path.exists(pygol_root) and os.path.exists(os.path.join(pygol_root, 'pygol.so')):
            return pygol_root
    for path in sys.path:
        if os.path.exists(path) and os.path.exists(os.path.join(path, 'pygol.so')):
            return path
    return None

_pygol_root = _find_pygol_root()
if _pygol_root:
    sys.path.insert(0, _pygol_root)


# Geometry1-specific learner - no external dataset config needed


class IterativePyGolZ3Learner(BaseEstimator, ClassifierMixin):
    """
    Iterative PyGol+Z3 Learner that refines rules through multiple iterations:
    1. PyGol proposes rules (ILP)
    2. Z3 SMT verifies and prunes invalid rules
    3. Verified rules are added to background knowledge
    4. PyGol learns again with enriched BK
    5. Repeat until convergence
    """
    
    def __init__(
        self,
        max_iterations: int = 10,
        convergence_threshold: float = 0.005,
        max_literals: int = 3,
        verbose: bool = True,
        bk_file: str = None,
        use_cross_validation: bool = False,
        cv_folds: int = 5,
        use_pareto_filtering: bool = False,
        pygol_timeout: int = 30
    ):
        """
        Parameters:
        -----------
        max_iterations : int
            Maximum number of refinement iterations
        convergence_threshold : float
            Minimum improvement in rule quality to continue
        max_literals : int
            Maximum literals per rule for PyGol
        verbose : bool
            Print progress information
        bk_file : str
            Background knowledge file path
        use_cross_validation : bool
            Enable cross-validation mode
        cv_folds : int
            Number of folds for cross-validation
        use_pareto_filtering : bool
            Enable Pareto-optimal filtering
        """
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
        self.max_literals = max_literals
        self.verbose = verbose
        if bk_file is None:
            import tempfile
            temp_dir = tempfile.gettempdir()
            self.bk_file = os.path.join(temp_dir, f'pygol_bk_{os.getpid()}.pl')
        else:
            self.bk_file = bk_file
        self.use_cross_validation = use_cross_validation
        self.cv_folds = cv_folds
        self.use_pareto_filtering = use_pareto_filtering
        
        self.ensemble_method = 'weighted_voting'
        self.use_feature_selection = False
        
        self.scaler = StandardScaler()
        self.feature_encoders = {}
        self.categorical_features = []
        self.learned_rules = []
        self.iteration_history = []
        self.blocking_constraints = []
        self.use_optimization = True
        self.soft_constraint_weight = 1.0
        self.pygol_timeout = pygol_timeout
        
    def _preprocess_data(self, X: pd.DataFrame):
        """Preprocess data: handle categorical features and scaling"""
        X_processed = X.copy()
        
        for col in X_processed.columns:
            if X_processed[col].dtype == 'object':
                numeric_series = pd.to_numeric(X_processed[col], errors='coerce')
                if not numeric_series.isna().all():
                    X_processed[col] = numeric_series
        
        for col in X_processed.columns:
            if X_processed[col].dtype == 'object' or (X_processed[col].dtype in ['float64', 'int64'] and X_processed[col].nunique() < 10):
                if col not in self.categorical_features:
                    self.categorical_features.append(col)
                    
        for col in self.categorical_features:
            if col not in self.feature_encoders:
                self.feature_encoders[col] = LabelEncoder()
                X_processed[col] = self.feature_encoders[col].fit_transform(X_processed[col].astype(str).fillna('missing'))
            else:
                X_processed[col] = self.feature_encoders[col].transform(X_processed[col].astype(str).fillna('missing'))
        
        numerical_cols = []
        for col in X_processed.columns:
            if col not in self.categorical_features:
                if pd.api.types.is_numeric_dtype(X_processed[col]):
                    numerical_cols.append(col)
                else:
                        try:
                            numeric_series = pd.to_numeric(
                                X_processed[col], errors='coerce')
                            if not numeric_series.isna().all():
                                X_processed[col] = numeric_series
                                numerical_cols.append(col)
                        except:
                            pass

        if numerical_cols:
                try:
                    X_processed[numerical_cols] = self.scaler.fit_transform(X_processed[numerical_cols])
                except Exception as e:
                    if self.verbose:
                        print(f"Warning: Could not scale numerical columns: {e}")
        
        return X_processed
    
    def _create_initial_background_knowledge(self, X: pd.DataFrame, y: np.ndarray, original_bk_file: Optional[str] = None):
        """Create initial background knowledge file from data"""
        with open(self.bk_file, 'w') as f:
            f.write("% Arithmetic operations and comparison predicates\n")
            f.write("magic(_).\n")
            f.write("geq(A,B) :- nonvar(A), nonvar(B), A>=B.\n")
            f.write("leq(A,B) :- nonvar(A), nonvar(B), A=<B.\n")
            f.write("eq(A,A) :- nonvar(A).\n")
            f.write("lt(A,B) :- nonvar(A), nonvar(B), A<B.\n")
            f.write("gt(A,B) :- nonvar(A), nonvar(B), A>B.\n")
            f.write("add(A,B,C) :- nonvar(A), nonvar(B), C is A+B.\n")
            f.write("add(A,B,C) :- nonvar(A), nonvar(C), B is C-A.\n")
            f.write("add(A,B,C) :- nonvar(B), nonvar(C), A is C-B.\n")
            f.write("mult(A,B,C) :- nonvar(A), nonvar(B), C is A*B.\n")
            f.write("mult(A,B,C) :- nonvar(A), nonvar(C), \\+(A=0.0), \\+(A=0), B is C/A.\n")
            f.write("mult(A,B,C) :- nonvar(B), nonvar(C), \\+(B=0.0), \\+(B=0), A is C/B.\n")
            f.write("\n")

            if original_bk_file and os.path.exists(original_bk_file):
                with open(original_bk_file, 'r') as orig_f:
                    for line in orig_f:
                        line = line.strip()
                        if any(pred in line for pred in ['my_add', 'my_mult', 'my_geq', 'my_leq']):
                            f.write(f"{line}\n")
                        elif line.startswith('%') or line.startswith(':-'):
                            continue

            skip_predicates = []
            if self.dataset_config and 'skip_predicates' in self.dataset_config:
                skip_predicates = self.dataset_config['skip_predicates']
            invalid_features = skip_predicates.copy() if skip_predicates else []

            for col in X.columns:
                if col not in invalid_features:
                    f.write(f":- dynamic {col}/2.\n")
            
            f.write("\n% Feature facts\n")
            for idx, (_, row) in enumerate(X.iterrows()):
                example_name = f"e_{idx + 1}"
                for col in X.columns:
                    if col in invalid_features:
                        continue

                    value = row[col]
                    if pd.isna(value):
                        value = 0.0

                    if col in self.categorical_features and col in self.feature_encoders:
                        try:
                            if pd.api.types.is_numeric_dtype(type(value)) or isinstance(value, (int, float)):
                                value_int = int(float(value))
                                encoder = self.feature_encoders[col]
                                if hasattr(encoder, 'classes_') and value_int < len(encoder.classes_):
                                    value = encoder.classes_[value_int]
                        except (ValueError, TypeError, IndexError):
                            pass

                    if pd.api.types.is_numeric_dtype(
    X[col]) and col not in self.categorical_features:
                        f.write(f"{col}({example_name}, {value}).\n")
                    else:
                        # Categorical or non-numeric: write as string
                        value_str = str(value).replace("'", "\\'")
                        f.write(f"{col}({example_name}, '{value_str}').\n")
    
            # For geometry problems, add example arithmetic facts to guide PyGol
            # These help PyGol see how arithmetic predicates can be used
            target_pred = getattr(self, 'target_predicate', 'target')
            if target_pred in ['halfplane', 'halfplane3d', 'conjunction', 'multihalfplane', 'interval', 'interval3d']:
                f.write("\n% Example arithmetic facts to guide PyGol structure learning\n")
                f.write("% These show PyGol how to use arithmetic predicates\n")
                # Add some example arithmetic combinations for each example
                # This helps PyGol see the pattern: x(X,V1), y(X,V2), add(V1,V2,Sum), leq(Sum,Threshold)
                for idx, (_, row) in enumerate(X.iterrows()):
                    example_name = f"e_{idx + 1}"
                    # Get feature values
                    if 'x' in X.columns and 'y' in X.columns:
                        try:
                            x_val = float(pd.to_numeric(row['x'], errors='coerce'))
                            y_val = float(pd.to_numeric(row['y'], errors='coerce'))
                            if not pd.isna(x_val) and not pd.isna(y_val):
                                sum_val = x_val + y_val
                                # Add example: add(x_val, y_val, sum_val)
                                f.write(f"add({x_val:.4f}, {y_val:.4f}, {sum_val:.4f}).\n")
                                # Add example comparisons
                                if sum_val > 0:
                                    f.write(f"geq({sum_val:.4f}, 0.0).\n")
                                if sum_val < 0:
                                    f.write(f"leq({sum_val:.4f}, 0.0).\n")
                        except:
                            pass
                    if 'x' in X.columns and 'y' in X.columns and 'z' in X.columns:
                        try:
                            x_val = float(pd.to_numeric(row['x'], errors='coerce'))
                            y_val = float(pd.to_numeric(row['y'], errors='coerce'))
                            z_val = float(pd.to_numeric(row['z'], errors='coerce'))
                            if not pd.isna(x_val) and not pd.isna(y_val) and not pd.isna(z_val):
                                # Example: 2*x + 3*y
                                mult1 = 2.0 * x_val
                                mult2 = 3.0 * y_val
                                sum_val = mult1 + mult2
                                f.write(f"mult({x_val:.4f}, 2.0, {mult1:.4f}).\n")
                                f.write(f"mult({y_val:.4f}, 3.0, {mult2:.4f}).\n")
                                f.write(f"add({mult1:.4f}, {mult2:.4f}, {sum_val:.4f}).\n")
                        except:
                            pass
                f.write("\n")
    
    def _add_verified_rules_to_bk(self, verified_rules: List[str]):
        """Add verified rules to background knowledge as constraints"""
        with open(self.bk_file, 'a') as f:
            f.write("\n% Verified rules from previous iteration\n")
            for rule in verified_rules:
                # Convert internal rule format to Prolog
                prolog_rule = self._rule_to_prolog(rule)
                if prolog_rule:
                    f.write(f"{prolog_rule}\n")
    
    def _rule_to_prolog(self, rule: Dict) -> Optional[str]:
        """Convert internal rule format to formal Prolog syntax using BK predicates (leq, geq, mult, add)
        
        This ensures rules are in the formal format compatible with background knowledge
        and can be used by PyGol in subsequent iterations.
        All numeric values are rounded to 2 decimal places for readability.
        """
        try:
            # Get target predicate name (default to 'target', but can be 'interval', 'halfplane', etc.)
            target_pred = getattr(self, 'target_predicate', 'target')
            
            # Helper function to format numbers to 2 decimal places
            def fmt(num):
                return round(float(num), 2)
            
            # Determine variable pattern based on target predicate arity
            # Use A, B for halfplane to match standard format: halfplane(A, B)
            # Use A, B, C for 3D halfplane: halfplane3d(A, B, C)
            if target_pred in ['halfplane3d']:
                head_pattern = f"{target_pred}(A, B, C)"
                var_x = 'A'
                var_y = 'B'
                var_z = 'C'
            elif target_pred in ['halfplane']:
                head_pattern = f"{target_pred}(A, B)"
                var_x = 'A'
                var_y = 'B'
                var_z = None
            elif target_pred in ['interval', 'interval3d']:
                if target_pred == 'interval3d':
                    head_pattern = f"{target_pred}(A, B, C)"
                    var_x = 'A'
                    var_y = 'B'
                    var_z = 'C'
                else:
                    head_pattern = f"{target_pred}(A)"
                    var_x = 'A'
                    var_y = None
                    var_z = None
            elif target_pred in ['conjunction']:
                head_pattern = f"{target_pred}(A, B, C)"
                var_x = 'A'
                var_y = 'B'
                var_z = 'C'
            else:
                head_pattern = f"{target_pred}(A)"
                var_x = 'A'
                var_y = None
                var_z = None
            
            if rule['type'] == 'single_feature':
                feature = rule['feature']
                operation = rule['operation']
                threshold = rule.get('threshold', 0)
                
                # For geometry problems, features are coordinates (x, y)
                if target_pred in ['halfplane', 'interval']:
                    if feature == 'x':
                        coord_var = var_x
                    elif feature == 'y' and var_y:
                        coord_var = var_y
                    else:
                        coord_var = var_x
                    
                    if operation == '>':
                        return f"{head_pattern} :- geq({coord_var}, {fmt(threshold)})."
                    elif operation == '>=':
                        return f"{head_pattern} :- geq({coord_var}, {fmt(threshold)})."
                    elif operation == '<':
                        return f"{head_pattern} :- leq({coord_var}, {fmt(threshold)})."
                    elif operation == '<=':
                        return f"{head_pattern} :- leq({coord_var}, {fmt(threshold)})."
                    elif operation == '==':
                        value = rule.get('value', threshold)
                        return f"{head_pattern} :- geq({coord_var}, {fmt(value)}), leq({coord_var}, {fmt(value)})."
                else:
                    # Generic format for other problems
                    if operation == '>':
                        return f"{head_pattern} :- {feature}(X, V), geq(V, {fmt(threshold)})."
                    elif operation == '<':
                        return f"{head_pattern} :- {feature}(X, V), leq(V, {fmt(threshold)})."
                    elif operation == '==':
                        value = rule.get('value', threshold)
                        return f"{head_pattern} :- {feature}(X, {fmt(value)})."
            
            elif rule['type'] == 'range':
                # Range rule: lower < x < upper
                feature = rule.get('feature', 'x')
                lower_bound = rule.get('lower_bound', 0)
                upper_bound = rule.get('upper_bound', 0)
                
                if target_pred in ['interval', 'halfplane']:
                    if feature == 'x':
                        coord_var = var_x
                    elif feature == 'y' and var_y:
                        coord_var = var_y
                    else:
                        coord_var = var_x
                    
                    return f"{head_pattern} :- geq({coord_var}, {fmt(lower_bound)}), leq({coord_var}, {fmt(upper_bound)})."
                else:
                    return f"{head_pattern} :- {feature}(X, V), geq(V, {fmt(lower_bound)}), leq(V, {fmt(upper_bound)})."
            
            elif rule['type'] == 'arithmetic_linear':
                # Arithmetic linear: coeff1*x + coeff2*y <= threshold (or 3-variable: coeff1*x + coeff2*y + coeff3*z <= threshold)
                features = rule.get('features', [])
                coefficients = rule.get('coefficients', [])
                threshold = rule.get('threshold', 0)
                operation = rule.get('operation', '<=')
                
                if len(features) == 3 and len(coefficients) == 3:
                    # 3-variable case: a*x + b*y + c*z <= threshold
                    col1, col2, col3 = features
                    coeff1, coeff2, coeff3 = coefficients
                    
                    if target_pred in ['halfplane3d']:
                        # For 3D halfplane: mult(x, a, D), mult(y, b, Temp1), add(Temp1, D, E), mult(z, c, Temp2), add(Temp2, E, F), leq(F, threshold)
                        if operation == '<=':
                            return f"{head_pattern} :- mult({var_x}, {fmt(coeff1)}, D), mult({var_y}, {fmt(coeff2)}, Temp1), add(Temp1, D, E), mult({var_z}, {fmt(coeff3)}, Temp2), add(Temp2, E, F), leq(F, {fmt(threshold)})."
                        elif operation == '>=':
                            return f"{head_pattern} :- mult({var_x}, {fmt(coeff1)}, D), mult({var_y}, {fmt(coeff2)}, Temp1), add(Temp1, D, E), mult({var_z}, {fmt(coeff3)}, Temp2), add(Temp2, E, F), geq(F, {fmt(threshold)})."
                    else:
                        # Generic 3-variable format
                        if operation == '<=':
                            return f"{head_pattern} :- {col1}(X, V1), {col2}(X, V2), {col3}(X, V3), mult(V1, {fmt(coeff1)}, Prod1), mult(V2, {fmt(coeff2)}, Prod2), mult(V3, {fmt(coeff3)}, Prod3), add(Prod1, Prod2, Sum1), add(Sum1, Prod3, Sum), leq(Sum, {fmt(threshold)})."
                        elif operation == '>=':
                            return f"{head_pattern} :- {col1}(X, V1), {col2}(X, V2), {col3}(X, V3), mult(V1, {fmt(coeff1)}, Prod1), mult(V2, {fmt(coeff2)}, Prod2), mult(V3, {fmt(coeff3)}, Prod3), add(Prod1, Prod2, Sum1), add(Sum1, Prod3, Sum), geq(Sum, {fmt(threshold)})."
                    return None
                
                if len(features) != 2 or len(coefficients) != 2:
                    return None
                
                col1, col2 = features
                coeff1, coeff2 = coefficients
                
                if target_pred in ['halfplane']:
                    if col1 == 'x' and col2 == 'y':
                        if abs(coeff2 - 1.0) < 0.01:
                            # Simple case: mult(x, coeff1, D), add(y, D, E), leq(E, threshold)
                            if operation == '<=':
                                return f"{head_pattern} :- mult({var_x}, {fmt(coeff1)}, D), add({var_y}, D, E), leq(E, {fmt(threshold)})."
                            elif operation == '>=':
                                return f"{head_pattern} :- mult({var_x}, {fmt(coeff1)}, D), add({var_y}, D, E), geq(E, {fmt(threshold)})."
                            elif operation == '<':
                                return f"{head_pattern} :- mult({var_x}, {fmt(coeff1)}, D), add({var_y}, D, E), leq(E, {fmt(threshold)})."
                            elif operation == '>':
                                return f"{head_pattern} :- mult({var_x}, {fmt(coeff1)}, D), add({var_y}, D, E), geq(E, {fmt(threshold)})."
                        else:
                            # General case: mult(x, coeff1, D), mult(y, coeff2, Temp), add(Temp, D, E), leq(E, threshold)
                            if operation == '<=':
                                return f"{head_pattern} :- mult({var_x}, {fmt(coeff1)}, D), mult({var_y}, {fmt(coeff2)}, Temp), add(Temp, D, E), leq(E, {fmt(threshold)})."
                            elif operation == '>=':
                                return f"{head_pattern} :- mult({var_x}, {fmt(coeff1)}, D), mult({var_y}, {fmt(coeff2)}, Temp), add(Temp, D, E), geq(E, {fmt(threshold)})."
                            elif operation == '<':
                                return f"{head_pattern} :- mult({var_x}, {fmt(coeff1)}, D), mult({var_y}, {fmt(coeff2)}, Temp), add(Temp, D, E), leq(E, {fmt(threshold)})."
                            elif operation == '>':
                                return f"{head_pattern} :- mult({var_x}, {fmt(coeff1)}, D), mult({var_y}, {fmt(coeff2)}, Temp), add(Temp, D, E), geq(E, {fmt(threshold)})."
                    else:
                        # Use feature predicates
                        # Convention: mult(Variable, Coefficient, Result)
                        if operation == '<=':
                            return f"{head_pattern} :- {col1}(X, Y, V1), {col2}(X, Y, V2), mult(V1, {fmt(coeff1)}, Prod1), mult(V2, {fmt(coeff2)}, Prod2), add(Prod1, Prod2, Sum), leq(Sum, {fmt(threshold)})."
                        elif operation == '>=':
                            return f"{head_pattern} :- {col1}(X, Y, V1), {col2}(X, Y, V2), mult(V1, {fmt(coeff1)}, Prod1), mult(V2, {fmt(coeff2)}, Prod2), add(Prod1, Prod2, Sum), geq(Sum, {fmt(threshold)})."
                else:
                    # Generic format
                    # Convention: mult(Variable, Coefficient, Result)
                    if operation == '<=':
                        return f"{head_pattern} :- {col1}(X, V1), {col2}(X, V2), mult(V1, {fmt(coeff1)}, Prod1), mult(V2, {fmt(coeff2)}, Prod2), add(Prod1, Prod2, Sum), leq(Sum, {fmt(threshold)})."
                    elif operation == '>=':
                        return f"{head_pattern} :- {col1}(X, V1), {col2}(X, V2), mult(V1, {fmt(coeff1)}, Prod1), mult(V2, {fmt(coeff2)}, Prod2), add(Prod1, Prod2, Sum), geq(Sum, {fmt(threshold)})."
            
            elif rule['type'] == 'conjunction':
                components = []
                for comp in rule['components']:
                    feature = comp['feature']
                    op = comp['operation']
                    threshold = comp.get('threshold', 0)
                    
                    if target_pred in ['halfplane', 'interval']:
                        if feature == 'x':
                            coord_var = var_x
                        elif feature == 'y' and var_y:
                            coord_var = var_y
                        else:
                            coord_var = var_x
                        
                        if op == '>':
                            components.append(f"geq({coord_var}, {fmt(threshold)})")
                        elif op == '>=':
                            components.append(f"geq({coord_var}, {fmt(threshold)})")
                        elif op == '<':
                            components.append(f"leq({coord_var}, {fmt(threshold)})")
                        elif op == '<=':
                            components.append(f"leq({coord_var}, {fmt(threshold)})")
                        elif op == '==':
                            value = comp.get('value', threshold)
                            components.append(f"geq({coord_var}, {fmt(value)}), leq({coord_var}, {fmt(value)})")
                    else:
                        if op == '>':
                            components.append(f"{feature}(X, V{len(components)}), geq(V{len(components)}, {fmt(threshold)})")
                        elif op == '<':
                            components.append(f"{feature}(X, V{len(components)}), leq(V{len(components)}, {fmt(threshold)})")
                        elif op == '==':
                            value = comp.get('value', threshold)
                            components.append(f"{feature}(X, {fmt(value)})")
                
                if components:
                    body = ", ".join(components)
                    return f"{head_pattern} :- {body}."
        except Exception as e:
            if self.verbose:
                print(f"Error converting rule to Prolog: {e}")
        return None
    
    def _prepare_pygol_examples(self, X: pd.DataFrame, y: np.ndarray):
        """Prepare positive and negative examples for PyGol"""
        # Create example files
        with open("pos_example.f", "w") as f:
            for idx, label in enumerate(y):
                if label == 1:
                    example_name = f"e_{idx + 1}"
                    f.write(f"target({example_name}).\n")
        
        with open("neg_example.n", "w") as f:
            for idx, label in enumerate(y):
                if label == 0:
                    example_name = f"e_{idx + 1}"
                    f.write(f"target({example_name}).\n")
        
        # Also return as dictionaries for direct use
        pos_examples = []
        neg_examples = []
        
        for idx, label in enumerate(y):
            example_name = f"e_{idx + 1}"
            if label == 1:
                pos_examples.append(example_name)
            else:
                neg_examples.append(example_name)
        
        Training_pos = {"example": pos_examples} if pos_examples else {}
        Training_neg = {"example": neg_examples} if neg_examples else {}
        
        return Training_pos, Training_neg
    
    def _extract_constants(self, X: pd.DataFrame) -> List:
        """Extract constants from data for PyGol"""
        constants = []
        
        for col in X.columns:
            try:
                if col in self.categorical_features:
                    # For categorical features, extract the original string values
                    # PyGol needs these constants to learn rules like color(X, 'red')
                    # We need to decode from numeric encodings back to strings
                    unique_vals = X[col].unique()
                    for v in unique_vals:
                        if pd.notna(v):
                            # Decode numeric encoding back to original string value
                            if col in self.feature_encoders:
                                try:
                                    if pd.api.types.is_numeric_dtype(type(v)) or isinstance(v, (int, float)):
                                        value_int = int(float(v))
                                        encoder = self.feature_encoders[col]
                                        if hasattr(encoder, 'classes_') and value_int < len(encoder.classes_):
                                            # Use original string value (e.g., 'red', 'blue')
                                            decoded_value = encoder.classes_[value_int]
                                            constants.append(str(decoded_value))
                                        else:
                                            constants.append(str(v))
                                    else:
                                        # Already a string, use as is
                                        constants.append(str(v))
                                except (ValueError, TypeError, IndexError):
                                    # If decoding fails, use the value as is
                                    constants.append(str(v))
                            else:
                                # No encoder, use value as is
                                constants.append(str(v))
                else:
                    # Use percentiles and key values as constants
                    target_pred = getattr(self, 'target_predicate', 'target')
                    if target_pred in ['halfplane', 'halfplane3d', 'conjunction', 'multihalfplane', 'interval', 'interval3d']:
                        percentiles = [10, 25, 50, 75, 90]
                        key_values = [0.0, -1.0, 1.0, -0.5, 0.5]
                    else:
                        percentiles = [25, 50, 75]
                        key_values = []
                    for pct in percentiles:
                        try:
                            val = X[col].quantile(pct / 100.0)
                            if pd.notna(val):
                                constants.append(str(val))
                        except Exception:
                            continue
                    
                    # Add key values if they're within the data range
                    try:
                        col_min = float(X[col].min())
                        col_max = float(X[col].max())
                        for key_val in key_values:
                            if col_min <= key_val <= col_max:
                                constants.append(str(key_val))
                    except Exception:
                        pass
            except Exception as e:
                # Skip columns that cause errors
                if self.verbose:
                    print(f"Warning: Could not extract constants from column {col}: {e}")
                continue

        # All constants are now strings, so sorting is safe
        try:
            unique_constants = sorted(list(set(constants)))
            if self.verbose:
                # Show categorical constants for debugging if any exist
                categorical_constants = [c for c in unique_constants if isinstance(c, str) and not c.startswith('e_')]
                if categorical_constants:
                    print(f"  [Constants] Extracted {len(categorical_constants)} categorical constants: {categorical_constants[:10]}")
            return unique_constants
        except Exception as e:
            # Fallback: return unsorted unique list
            if self.verbose:
                print(f"Warning: Could not sort constants: {e}")
            return list(set(constants))

    def _learn_rules_with_pygol(
    self,
    X: pd.DataFrame,
    y: np.ndarray,
     iteration: int = 0):
        """Learn rules using PyGol"""
        if self.verbose:
            print(f"\n[Iteration {iteration}] Learning rules with PyGol...")
        
        # Prepare examples
        Training_pos, Training_neg = self._prepare_pygol_examples(X, y)
        
        if not Training_pos or not Training_neg:
            if self.verbose:
                print("Warning: Insufficient examples")
            return []
        
        # Extract constants
        constants = self._extract_constants(X)
        
        # Ensure BK file path is absolute
        bk_file_path = os.path.abspath(self.bk_file)
        pos_example_path = os.path.abspath("pos_example.f")
        neg_example_path = os.path.abspath("neg_example.n")
        
        target_pred = getattr(self, 'target_predicate', 'target')
        if target_pred in ['halfplane', 'halfplane3d', 'conjunction', 'multihalfplane', 'interval', 'interval3d']:
            bottom_clause_depth = 4
        else:
            bottom_clause_depth = 2
        
        try:
            P, N = bottom_clause_generation(
                file=bk_file_path,
                constant_set=constants,
                depth=bottom_clause_depth,
                container="dict",
                positive_example=pos_example_path,
                negative_example=neg_example_path,
                positive_file_dictionary="positive_bottom_clause",
                negative_file_dictionary="negative_bottom_clause"
            )
            
            Train_P, Test_P, Train_N, Test_N = pygol_train_test_split(
                test_size=0,
                positive_file_dictionary=P,
                negative_file_dictionary=N
            )
        except Exception as e:
            if self.verbose:
                print(
    f"Bottom clause generation failed, using direct examples: {e}")
            Train_P = Training_pos
            Train_N = Training_neg
        
        import threading

        pygol_model = None
        pygol_error = [None]

        def run_pygol():
            try:
                nonlocal pygol_model
                target_pred = getattr(self, 'target_predicate', 'target')
                if target_pred in ['halfplane', 'halfplane3d', 'conjunction', 'multihalfplane', 'interval', 'interval3d']:
                    effective_max_literals = max(self.max_literals, 6)
                else:
                    effective_max_literals = self.max_literals
                
                pygol_model = pygol_learn(
            Training_pos=Train_P,
            Training_neg=Train_N,
            file=bk_file_path,
            constant_set=constants,
            max_literals=effective_max_literals,
            max_neg=0,
            min_pos=1,
            key_size=1,
            verbose=self.verbose
        )
            except Exception as e:
                pygol_error[0] = e

        target_pred = getattr(self, 'target_predicate', 'target')
        if target_pred in ['halfplane', 'halfplane3d', 'conjunction', 'multihalfplane', 'interval', 'interval3d']:
            effective_timeout = max(self.pygol_timeout, 120)
        else:
            effective_timeout = self.pygol_timeout
        
        pygol_thread = threading.Thread(target=run_pygol, daemon=True)
        pygol_thread.start()
        pygol_thread.join(timeout=effective_timeout)

        if pygol_thread.is_alive():
            if self.verbose:
                print(f"PyGol learning timed out after {effective_timeout} seconds (iteration {iteration})")
            return []  # Return empty rules if timeout

        if pygol_error[0]:
            if self.verbose:
                print(f"Error in PyGol learning: {pygol_error[0]}")
            return []

        if not hasattr(
    pygol_model,
     'hypothesis') or not pygol_model.hypothesis:
            if self.verbose:
                print("No rules learned from PyGol")
            return []
        
        if self.verbose:
            print(f"PyGol learned {len(pygol_model.hypothesis)} rules")
            for rule in pygol_model.hypothesis:
                print(f"  - {rule}")
        
        # Convert to internal format
        return self._convert_pygol_rules(pygol_model.hypothesis, X)
    
    def _convert_pygol_rules(
    self,
    pygol_rules: List[str],
     X: pd.DataFrame) -> List[Dict]:
        """
        Global structure induction using PyGOL MIE
        Convert PyGol Prolog rules to internal format with numeric placeholders.
        Extract candidate clauses with thresholds that will be optimized by SMT.
        """
        converted = []
        
        for rule_str in pygol_rules:
            try:
                if ':-' not in rule_str:
                    continue
                
                body = rule_str.split(':-')[1].strip().rstrip('.')
                components = []
                
                # Track feature variables and their comparisons
                feature_vars = {}  # Maps variable name to feature name
                
                # Reconstruct full body to handle predicates split by commas
                # Look for patterns like: feature_0(X, V) or feature_0(X, 1.0)
                
                # First pass: find all feature predicates with their variables
                skip_rule = False  # Flag to skip rules with categorical variables
                for col in X.columns:
                    # Pattern: feature_0(X, V) or feature_0(X, 1.0) or feature_0(X, 'red')
                    pattern = f"{re.escape(col)}\\([^,]+,\\s*([^)]+)\\)"
                    matches = re.finditer(pattern, body)
                    for match in matches:
                        var_or_value = match.group(1).strip()
                        # Check if it's a variable (starts with letter, not quoted) or value (number/string)
                        if var_or_value[0].isalpha() and not (var_or_value.startswith("'") or var_or_value.startswith('"')):
                            # It's a variable (e.g., B, V, C)
                            # For categorical features, variables make rules too general
                            # BUT: Don't skip the rule - mark it for post-processing instead
                            # We'll use _add_categorical_constraints to replace variables with specific values
                            if col in self.categorical_features:
                                # Categorical feature as variable - mark it for post-processing
                                # Store the variable name so we can replace it later
                                if 'categorical_variables' not in locals():
                                    categorical_variables = {}
                                categorical_variables[col] = var_or_value
                                # Add placeholder component for categorical constraints
                                # Mark it with a special operation 'variable' so we know it needs to be replaced
                                components.append({
                                    'feature': col,
                                    'operation': 'variable',  # Special marker for categorical variable
                                    'variable_name': var_or_value,
                                    'threshold': None,
                                    'value': None
                                })
                                # Don't add to feature_vars - we'll handle it in post-processing
                                if self.verbose:
                                    print(f"  [Rule Conversion] Marking categorical variable for post-processing: {col}(X, {var_or_value})")
                            else:
                                # Numeric feature as variable is OK (will be constrained by comparison)
                                feature_vars[var_or_value] = col
                        else:
                            # It's a constant value (number or quoted string)
                            try:
                                # Try numeric first
                                value = float(var_or_value)
                                # Convert exact matches to threshold rules for better generalization
                                # Exact matches like y(A, -0.716) are too specific and don't generalize
                                # Instead, convert to threshold rules that will be optimized by SMT
                                # Use the value as an initial threshold, but mark it as optimizable
                                # We'll use '>' or '<' operation and let SMT optimize the threshold
                                # For geometry problems, exact matches are almost never what we want
                                if col not in self.categorical_features:
                                    # For numeric features: convert exact match to threshold rule
                                    # Use the value as initial threshold, operation will be optimized by SMT
                                    components.append({
                                        'feature': col,
                                        'operation': '>',  # Default to '>', SMT will optimize
                                        'threshold': value,
                                        'threshold_var': None,
                                        'has_optimizable_threshold': True,  # Mark as optimizable
                                        'from_exact_match': True  # Track that this came from exact match
                                    })
                                else:
                                    # For categorical features: keep exact match
                                    components.append({
                                        'feature': col,
                                        'operation': '==',
                                        'value': value,
                                        'threshold_var': None
                                    })
                            except:
                                # String value (remove quotes)
                                value = var_or_value.strip("'\"")
                                components.append({
                                    'feature': col,
                                    'operation': '==',
                                    'value': value,
                                    'threshold_var': None
                                })
                
                # Skip rules with categorical variables (they're too general)
                if skip_rule:
                    continue
                
                # Second pass: find comparisons using tracked variables
                # Pattern: V > 0.5 or V < -0.5
                for var_name, feature in feature_vars.items():
                    # Look for: var_name > threshold or var_name < threshold
                    pattern_greater = f"\\b{re.escape(var_name)}\\s*>\\s*([0-9.-]+)"
                    pattern_less = f"\\b{re.escape(var_name)}\\s*<\\s*([0-9.-]+)"
                    
                    match_greater = re.search(pattern_greater, body)
                    if match_greater:
                        try:
                            threshold = float(match_greater.group(1))
                            components.append({
                                'feature': feature,
                                'operation': '>',
                                'threshold': threshold,
                                'threshold_var': None
                            })
                        except:
                            pass
                    
                    match_less = re.search(pattern_less, body)
                    if match_less:
                        try:
                            threshold = float(match_less.group(1))
                            components.append({
                                'feature': feature,
                                'operation': '<',
                                'threshold': threshold,
                                'threshold_var': None
                            })
                        except:
                            pass
                
                # Third pass: Handle arithmetic predicates (sum_x_y, scaled_2_x, add_fact, etc.)
                # These are learned as facts with specific values, which we convert to threshold rules
                # Pattern: sum_x_y(A, 3.5) -> treat as sum_x_y(A, V), V > threshold
                # Pattern: scaled_2_x(A, 1.4) -> treat as scaled_2_x(A, V), V >
                # threshold
                arithmetic_patterns = [
                    r'sum_([^_]+)_([^_]+)\([^,]+,\s*([0-9.-]+)\)',
                    r'prod_([^_]+)_([^_]+)\([^,]+,\s*([0-9.-]+)\)',
                    r'diff_([^_]+)_([^_]+)\([^,]+,\s*([0-9.-]+)\)',
                    r'scaled_([0-9.]+)_([^_]+)\([^,]+,\s*([0-9.-]+)\)',
                    r'add_fact\([^,]+,\s*[^,]+,\s*[^,]+,\s*([0-9.-]+)\)',
                    r'mult_fact\([^,]+,\s*[^,]+,\s*[^,]+,\s*([0-9.-]+)\)',
                ]

                for pattern in arithmetic_patterns:
                    matches = re.finditer(pattern, body)
                    for match in matches:
                        try:
                            # Extract the value (last group is always the
                            # numeric value)
                            value = float(match.group(match.lastindex))

                            # Determine the feature name based on the pattern
                            if 'sum_' in pattern or 'prod_' in pattern or 'diff_' in pattern:
                                # For sum_x_y, prod_x_y, diff_x_y: use the
                                # combined name
                                feature_name = match.group(0).split(
                                    '(')[0]  # e.g., "sum_x_y"
                            elif 'scaled_' in pattern:
                                # For scaled_2_x: use the full name
                                feature_name = match.group(0).split(
                                    '(')[0]  # e.g., "scaled_2_x"
                            elif 'add_fact' in pattern or 'mult_fact' in pattern:
                                # For add_fact/mult_fact: use the predicate
                                # name
                                feature_name = match.group(0).split(
                                    '(')[0]  # e.g., "add_fact"
                            else:
                                continue

                            # Convert to a threshold rule: treat the value as an initial threshold
                            # We'll use '>' operation and let SMT optimize the
                            # threshold
                            components.append({
                                'feature': feature_name,
                                'operation': '>',  # Default to >, SMT will optimize
                                'threshold': value,
                                'threshold_var': None,
                                'is_arithmetic': True  # Mark as arithmetic predicate
                            })
                        except (ValueError, IndexError):
                            continue
                
                if components:
                    if len(components) == 1:
                        rule = {
                            'type': 'single_feature',
                            'feature': components[0]['feature'],
                            **components[0],
                            'pygol_source': rule_str,
                            'has_optimizable_threshold': components[0].get('operation') in ['>', '<']
                        }
                    else:
                        rule = {
                            'type': 'conjunction',
                            'components': components,
                            'pygol_source': rule_str,
                            'has_optimizable_threshold': any(c.get('operation') in ['>', '<'] for c in components)
                        }
                    
                    # Mark if rule has categorical variables that need post-processing
                    if 'categorical_variables' in locals() and categorical_variables:
                        rule['has_categorical_variables'] = True
                        rule['categorical_variables'] = categorical_variables.copy()
                    
                    converted.append(rule)
            except Exception as e:
                if self.verbose:
                    print(f"Error parsing rule '{rule_str}': {e}")
                continue
        
        return converted
    
    def _optimize_threshold_smt(self, rule: Dict, X: pd.DataFrame, y: np.ndarray, 
                                feature: str, operation: str, initial_threshold: float) -> Optional[float]:
        """
        SMT fitting with Optimize() for threshold optimization
        Find optimal threshold value that maximizes positive coverage and minimizes negative coverage
        Uses Z3 Optimize() to explicitly optimize objectives rather than just checking satisfiability
        """
        y_series = pd.Series(y, index=X.index)
        pos_indices = X.index[y_series == 1].tolist()
        neg_indices = X.index[y_series == 0].tolist()
        
        if len(pos_indices) == 0 or len(neg_indices) == 0:
            return initial_threshold
        
        # Use Optimize() instead of Solver() for proper optimization
        if self.use_optimization:
            opt = Optimize()
        else:
            opt = Solver()  # Fallback to Solver if optimization disabled
        
        threshold = Real('threshold')
        
        # Set bounds based on data range (only for numeric features)
        # Always convert to numeric first to handle mixed types safely
        try:
            feature_series = pd.to_numeric(X[feature], errors='coerce')
            # Force dtype to float to avoid object dtype issues
            # This ensures numpy can do comparisons even if original column was
            # mixed types
            feature_series = feature_series.astype(float)
            # Check if we got any valid numeric values
            if feature_series.isna().all():
                # All values were non-numeric, skip threshold optimization
                if self.verbose:
                    print(
    f"Skipping threshold optimization for non-numeric feature: {feature}")
                return None

            feat_min = float(feature_series.min())
            feat_max = float(feature_series.max())
            opt.add(threshold >= feat_min)
            opt.add(threshold <= feat_max)
        except (TypeError, ValueError) as e:
            # If conversion fails, skip threshold optimization
            if self.verbose:
                print(
    f"Skipping threshold optimization for feature {feature}: {e}")
            return None
        
        # Apply blocking constraints to prevent similar bad rules
        self._apply_blocking_constraints(opt, feature, threshold, operation)
        
        # Create SMT variables for each example
        # Count how many positives and negatives satisfy the rule
        pos_satisfied = []
        neg_satisfied = []
        neg_soft_constraints = []  # Store soft constraints for negatives
        
        for idx in pos_indices:
            feat_val = Real(f'pos_{idx}_{feature}')
            # Ensure value is numeric before converting
            val = X.loc[idx, feature]
            # Try to convert to float, skip if it fails
            try:
                val_float = float(pd.to_numeric(val, errors='coerce'))
                if pd.isna(val_float):
                    continue  # Skip NaN values (non-numeric)
                opt.add(feat_val == val_float)
            except (TypeError, ValueError):
                continue  # Skip non-numeric values
            if operation == '>':
                pos_satisfied.append(If(feat_val > threshold, 1, 0))
            elif operation == '<':
                pos_satisfied.append(If(feat_val < threshold, 1, 0))
        
        for idx in neg_indices:
            feat_val = Real(f'neg_{idx}_{feature}')
            # Ensure value is numeric before converting
            val = X.loc[idx, feature]
            # Try to convert to float, skip if it fails
            try:
                val_float = float(pd.to_numeric(val, errors='coerce'))
                if pd.isna(val_float):
                    continue  # Skip NaN values (non-numeric)
                opt.add(feat_val == val_float)
            except (TypeError, ValueError):
                continue  # Skip non-numeric values
            if operation == '>':
                neg_satisfied.append(If(feat_val > threshold, 1, 0))
                # Soft constraint - prefer that negative does NOT
                # satisfy
                neg_soft_constraints.append(feat_val <= threshold)
            elif operation == '<':
                neg_satisfied.append(If(feat_val < threshold, 1, 0))
                # Soft constraint - prefer that negative does NOT
                # satisfy
                neg_soft_constraints.append(feat_val >= threshold)
        
        # Explicit optimization objectives with soft constraints for negative examples
        if self.use_optimization and pos_satisfied and neg_satisfied:
            pos_count = Sum(pos_satisfied) if len(
                pos_satisfied) > 1 else pos_satisfied[0]
            neg_count = Sum(neg_satisfied) if len(
                neg_satisfied) > 1 else neg_satisfied[0]
            
            # Maximize positive coverage, minimize negative coverage
            opt.maximize(pos_count)
            opt.minimize(neg_count)
            
            # Add soft constraints for negatives (prefer but don't require exclusion)
            # This allows some negatives to be covered (noise tolerance)
            for soft_constraint in neg_soft_constraints:
                opt.add_soft(
    soft_constraint,
     weight=self.soft_constraint_weight)
        
        result = opt.check()
        if result == sat:
            model = opt.model()
            try:
                optimized = float(model[threshold].as_fraction())
                return float(optimized)
            except:
                # Fallback: try as_decimal()
                try:
                    optimized = float(model[threshold].as_decimal(10))
                    return float(optimized)
                except:
                    return float(
                        initial_threshold) if initial_threshold is not None else None
        else:
            return float(
                initial_threshold) if initial_threshold is not None else None

    def _learn_arithmetic_relationships_smt(
    self,
    rules: List[Dict],
    X: pd.DataFrame,
     y: np.ndarray) -> List[Dict]:
        """
        Learn arithmetic relationships from simple PyGol rules using SMT

        This method uses SMT to learn arithmetic relationships (like numsynth/Popper does).
        Instead of relying on PyGol to learn rules with arithmetic predicates, we:
        1. Let PyGol learn simple rules (as it naturally does)
        2. Use SMT to learn linear combinations: a*x + b*y <= threshold

        This is how numsynth/Popper works - it generates rules with arithmetic predicates
        and uses Z3 to find the numerical values.
        """
        if not self.use_optimization:
            return []

        arithmetic_rules = []
        # Ensure y and X.index have matching length
        # If they don't match, align them by using X.index length
        if len(y) != len(X.index):
            if self.verbose:
                print(f"  [Arithmetic Learning] Warning: y length ({len(y)}) != X.index length ({len(X.index)}), aligning...")
            # Use X.index length and align y accordingly
            if len(y) > len(X.index):
                y = y[:len(X.index)]
            else:
                # If y is shorter, pad with zeros (shouldn't happen, but handle it)
                y_padded = np.zeros(len(X.index))
                y_padded[:len(y)] = y
                y = y_padded
        
        y_series = pd.Series(y, index=X.index)
        pos_indices = X.index[y_series == 1].tolist()
        neg_indices = X.index[y_series == 0].tolist()

        if len(pos_indices) == 0 or len(neg_indices) == 0:
            return []

        # Get numeric columns
        numeric_cols = [
    col for col in X.columns if pd.api.types.is_numeric_dtype(
        X[col])]
        if len(numeric_cols) < 2:
            return []  # Need at least 2 numeric features for arithmetic

        if self.verbose:
            print(f"\n[Arithmetic Learning] Attempting to learn linear relationships from {len(numeric_cols)} numeric features...")

        # For multihalfplane problems, we need to find MULTIPLE arithmetic rules for the same feature pair
        # Multiple Halfplanes requires: (a1*x + b1*y <= c1) AND (a2*x + b2*y <= c2)
        # So we need to find 2+ distinct rules for the same (x, y) combination
        target_pred = getattr(self, 'target_predicate', None)
        is_multihalfplane = (target_pred == 'multihalfplane')
        max_rules_per_pair = 2 if is_multihalfplane else 1  # Find 2 rules for multihalfplane, 1 for others
        
        # IMPROVED: Use Z3 to find BOTH coefficients AND thresholds (like numsynth does)
        # This is more principled than trying fixed coefficient combinations
        # Try 2-variable combinations first
        for i, col1 in enumerate(numeric_cols):
            for j, col2 in enumerate(numeric_cols):
                if i >= j:  # Avoid duplicates
                    continue
                
                # For multihalfplane, try to find multiple rules for this feature pair
                rules_found_for_pair = 0
                covered_positives = set()  # Track which positives are covered by previous rules

                # Use Z3 to find optimal coefficients AND threshold
                # simultaneously
                opt = Optimize()
                coeff1 = Real('coeff1')
                coeff2 = Real('coeff2')
                threshold = Real('threshold')

                # Constrain coefficients to reasonable ranges (like numsynth's bounds)
                # Use dataset-specific configuration from dataset_config
                dataset_config = getattr(self, 'dataset_config', None)
                if dataset_config and dataset_config.get('arithmetic_bounds'):
                    MIN_COEFF, MAX_COEFF = dataset_config['arithmetic_bounds']
                else:
                    # Fallback: use target predicate
                    target_pred = getattr(self, 'target_predicate', None)
                    if target_pred in ['halfplane', 'interval', 'target']:
                        # Geometry: bounds(-100, 100) for int
                        MIN_COEFF, MAX_COEFF = -100.0, 100.0
                    else:
                        # Default: allow reasonable range
                        MIN_COEFF, MAX_COEFF = -10.0, 10.0

                opt.add(coeff1 >= MIN_COEFF, coeff1 <= MAX_COEFF)
                opt.add(coeff2 >= MIN_COEFF, coeff2 <= MAX_COEFF)
                opt.add(Or(coeff1 != 0, coeff2 != 0))

                try:
                    val1_series = pd.to_numeric(
    X[col1], errors='coerce').astype(float)
                    val2_series = pd.to_numeric(
    X[col2], errors='coerce').astype(float)

                    if val1_series.isna().all() or val2_series.isna().all():
                        continue

                    val1_min = float(val1_series.min())
                    val1_max = float(val1_series.max())
                    val2_min = float(val2_series.min())
                    val2_max = float(val2_series.max())

                    max_linear = MAX_COEFF * max(abs(val1_min), abs(val1_max)) + MAX_COEFF * max(abs(val2_min), abs(val2_max))
                    opt.add(threshold >= -max_linear)
                    opt.add(threshold <= max_linear)

                    pos_satisfied = []
                    neg_soft_constraints = []

                    for idx in pos_indices:
                        v1 = float(pd.to_numeric(
                            X.loc[idx, col1], errors='coerce'))
                        v2 = float(pd.to_numeric(
                            X.loc[idx, col2], errors='coerce'))
                        if pd.isna(v1) or pd.isna(v2):
                            continue

                        linear_val = Real(f'pos_{idx}_lin')
                        opt.add(linear_val == coeff1 * v1 + coeff2 * v2)
                        pos_satisfied.append(If(linear_val <= threshold, 1, 0))

                    for idx in neg_indices:
                        v1 = float(pd.to_numeric(
                            X.loc[idx, col1], errors='coerce'))
                        v2 = float(pd.to_numeric(
                            X.loc[idx, col2], errors='coerce'))
                        if pd.isna(v1) or pd.isna(v2):
                            continue

                        linear_val = Real(f'neg_{idx}_lin')
                        opt.add(linear_val == coeff1 * v1 + coeff2 * v2)
                        neg_soft_constraints.append(linear_val > threshold)

                    if not pos_satisfied:
                        continue

                    opt.maximize(Sum(pos_satisfied))

                    for neg_constraint in neg_soft_constraints:
                        opt.add_soft(neg_constraint, weight=1)

                    if opt.check() == sat:
                        model = opt.model()
                        coeff1_val = float(model[coeff1].as_fraction())
                        coeff2_val = float(model[coeff2].as_fraction())
                        threshold_val = float(model[threshold].as_fraction())
                        
                        best_coeffs = [coeff1_val, coeff2_val]
                        best_threshold = threshold_val
                        best_precision = 0
                        best_pos_coverage = 0
                        
                        for refine_iter in range(5):
                            refine_opt = Optimize()
                            refine_coeff1 = Real('refine_coeff1')
                            refine_coeff2 = Real('refine_coeff2')
                            refine_threshold = Real('refine_threshold')
                            
                            # Tighter bounds: ±20% of current values, but still within original bounds
                            refine_range = 0.2  # 20% refinement range
                            refine_opt.add(refine_coeff1 >= max(MIN_COEFF, best_coeffs[0] * (1 - refine_range)))
                            refine_opt.add(refine_coeff1 <= min(MAX_COEFF, best_coeffs[0] * (1 + refine_range)))
                            refine_opt.add(refine_coeff2 >= max(MIN_COEFF, best_coeffs[1] * (1 - refine_range)))
                            refine_opt.add(refine_coeff2 <= min(MAX_COEFF, best_coeffs[1] * (1 + refine_range)))
                            refine_opt.add(refine_threshold >= best_threshold * (1 - refine_range))
                            refine_opt.add(refine_threshold <= best_threshold * (1 + refine_range))
                            
                            # Rebuild constraints with refined variables
                            refine_pos_satisfied = []
                            refine_neg_constraints = []
                            
                            for idx in pos_indices:
                                v1 = float(pd.to_numeric(X.loc[idx, col1], errors='coerce'))
                                v2 = float(pd.to_numeric(X.loc[idx, col2], errors='coerce'))
                                if pd.isna(v1) or pd.isna(v2):
                                    continue
                                
                                linear_val = Real(f'refine_pos_{idx}_lin')
                                refine_opt.add(linear_val == refine_coeff1 * v1 + refine_coeff2 * v2)
                                refine_pos_satisfied.append(If(linear_val <= refine_threshold, 1, 0))
                            
                            for idx in neg_indices:
                                v1 = float(pd.to_numeric(X.loc[idx, col1], errors='coerce'))
                                v2 = float(pd.to_numeric(X.loc[idx, col2], errors='coerce'))
                                if pd.isna(v1) or pd.isna(v2):
                                    continue
                                
                                linear_val = Real(f'refine_neg_{idx}_lin')
                                refine_opt.add(linear_val == refine_coeff1 * v1 + refine_coeff2 * v2)
                                refine_neg_constraints.append(linear_val > refine_threshold)
                            
                            if refine_pos_satisfied:
                                refine_opt.maximize(Sum(refine_pos_satisfied))
                                for neg_constraint in refine_neg_constraints:
                                    refine_opt.add_soft(neg_constraint, weight=1)
                                
                                if refine_opt.check() == sat:
                                    refine_model = refine_opt.model()
                                    refine_coeff1_val = float(refine_model[refine_coeff1].as_fraction())
                                    refine_coeff2_val = float(refine_model[refine_coeff2].as_fraction())
                                    refine_threshold_val = float(refine_model[refine_threshold].as_fraction())
                                    
                                    # Evaluate refined rule
                                    refine_linear_comb = refine_coeff1_val * val1_series + refine_coeff2_val * val2_series
                                    refine_mask = refine_linear_comb <= refine_threshold_val
                                    
                                    refine_pos_covered = refine_mask[y_series == 1].sum()
                                    refine_neg_covered = refine_mask[y_series == 0].sum()
                                    refine_total_pos = (y_series == 1).sum()
                                    
                                    refine_pos_coverage = refine_pos_covered / max(refine_total_pos, 1)
                                    refine_precision = refine_pos_covered / max(refine_mask.sum(), 1) if refine_mask.sum() > 0 else 0
                                    
                                    # Keep if better
                                    if refine_precision > best_precision or (refine_precision == best_precision and refine_pos_coverage > best_pos_coverage):
                                        best_coeffs = [refine_coeff1_val, refine_coeff2_val]
                                        best_threshold = refine_threshold_val
                                        best_precision = refine_precision
                                        best_pos_coverage = refine_pos_coverage
                        
                        # Use best coefficients from refinement
                        coeff1_val, coeff2_val = best_coeffs
                        threshold_val = best_threshold

                        # Evaluate the rule with learned coefficients
                        linear_comb_series = coeff1_val * val1_series + coeff2_val * val2_series
                        mask = linear_comb_series <= threshold_val

                        pos_covered = mask[y_series == 1].sum()
                        neg_covered = mask[y_series == 0].sum()
                        total_pos = (y_series == 1).sum()
                        total_neg = (y_series == 0).sum()

                        pos_coverage = pos_covered / max(total_pos, 1)
                        neg_exclusion = 1 - (neg_covered / max(total_neg, 1))
                        precision = pos_covered / \
                            max(mask.sum(), 1) if mask.sum() > 0 else 0
                        recall = pos_coverage  # For arithmetic rules, recall = pos_coverage
                        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

                        # Only keep if it's better than random (precision >
                        # 0.5)
                        if precision > 0.5 and pos_coverage > 0.3:
                            # For multihalfplane, check if we already have enough rules for this pair
                            if is_multihalfplane and rules_found_for_pair >= max_rules_per_pair:
                                continue  # Skip if we already have 2 rules for this pair
                            
                            arithmetic_rule = {
                                'type': 'arithmetic_linear',
                                'features': [col1, col2],
                                # Learned by Z3
                                'coefficients': [coeff1_val, coeff2_val],
                                'threshold': threshold_val,  # Learned by Z3
                                'operation': '<=',
                                'pos_coverage': pos_coverage,
                                'neg_exclusion': neg_exclusion,
                                'precision': precision,
                                'recall': recall,
                                'f1': f1,
                                'support': mask.sum() / len(X),
                                'score': 0.4 * pos_coverage + 0.3 * neg_exclusion + 0.2 * precision,
                                'verified': True,
                                'smt_optimized': True
                            }
                            arithmetic_rules.append(arithmetic_rule)
                            rules_found_for_pair += 1
                            
                            # Track covered positives for multihalfplane
                            if is_multihalfplane:
                                current_covered = set(idx for idx in pos_indices if mask.loc[idx])
                                covered_positives.update(current_covered)

                            if self.verbose:
                                print(f"  [OK] Learned: {coeff1_val:.4f}*{col1} + {coeff2_val:.4f}*{col2} <= {threshold_val:.4f} "
                                      f"(precision={precision:.2f}, coverage={pos_coverage:.2f})")
                            
                            # For multihalfplane, try to find a second rule
                            # Even if first rule covers all positives, we need a second constraint
                            # So we'll find a rule that's different (different coefficients/threshold)
                            if is_multihalfplane and rules_found_for_pair < max_rules_per_pair:
                                # Try to find a second rule with different coefficients
                                # Use all positives but with a constraint that coefficients must be different
                                # We'll try to find a rule that's complementary (covers similar positives but with different constraint)
                                remaining_pos_indices = [idx for idx in pos_indices if idx not in covered_positives]
                                # If first rule covers all positives, use all positives for second rule too
                                # The key is that the second rule must have different coefficients/threshold
                                if len(remaining_pos_indices) < 3:
                                    remaining_pos_indices = pos_indices  # Use all positives if none remaining
                                
                                if len(remaining_pos_indices) >= 3:  # Need at least 3 positives for a rule
                                    # Create a new optimizer for the second rule
                                    opt2 = Optimize()
                                    coeff1_2 = Real('coeff1_2')
                                    coeff2_2 = Real('coeff2_2')
                                    threshold_2 = Real('threshold_2')
                                    
                                    opt2.add(coeff1_2 >= MIN_COEFF, coeff1_2 <= MAX_COEFF)
                                    opt2.add(coeff2_2 >= MIN_COEFF, coeff2_2 <= MAX_COEFF)
                                    opt2.add(Or(coeff1_2 != 0, coeff2_2 != 0))
                                    opt2.add(threshold_2 >= -max_linear)
                                    opt2.add(threshold_2 <= max_linear)
                                    
                                    # Ensure second rule is different from first rule
                                    # Require that at least one coefficient is significantly different (at least 30% difference)
                                    # This ensures we get two distinct constraints
                                    # Z3 doesn't support abs() directly, so use If-then-else or separate constraints
                                    coeff1_diff = abs(coeff1_val) * 0.3 if abs(coeff1_val) > 0.1 else 0.1
                                    coeff2_diff = abs(coeff2_val) * 0.3 if abs(coeff2_val) > 0.1 else 0.1
                                    threshold_diff = abs(threshold_val) * 0.3 if abs(threshold_val) > 0.1 else 0.1
                                    
                                    # Require that at least one parameter is different
                                    # Use If to handle absolute value: abs(x) = If(x >= 0, x, -x)
                                    diff_coeff1_pos = If(coeff1_2 >= coeff1_val, coeff1_2 - coeff1_val, coeff1_val - coeff1_2)
                                    diff_coeff2_pos = If(coeff2_2 >= coeff2_val, coeff2_2 - coeff2_val, coeff2_val - coeff2_2)
                                    diff_threshold_pos = If(threshold_2 >= threshold_val, threshold_2 - threshold_val, threshold_val - threshold_2)
                                    
                                    opt2.add(Or(
                                        diff_coeff1_pos >= coeff1_diff,
                                        diff_coeff2_pos >= coeff2_diff,
                                        diff_threshold_pos >= threshold_diff
                                    ))
                                    
                                    pos_satisfied_2 = []
                                    neg_soft_constraints_2 = []
                                    
                                    # Use remaining positives
                                    for idx in remaining_pos_indices:
                                        v1 = float(pd.to_numeric(X.loc[idx, col1], errors='coerce'))
                                        v2 = float(pd.to_numeric(X.loc[idx, col2], errors='coerce'))
                                        if pd.isna(v1) or pd.isna(v2):
                                            continue
                                        linear_val = Real(f'pos2_{idx}_lin')
                                        opt2.add(linear_val == coeff1_2 * v1 + coeff2_2 * v2)
                                        pos_satisfied_2.append(If(linear_val <= threshold_2, 1, 0))
                                    
                                    # Still use all negatives
                                    for idx in neg_indices:
                                        v1 = float(pd.to_numeric(X.loc[idx, col1], errors='coerce'))
                                        v2 = float(pd.to_numeric(X.loc[idx, col2], errors='coerce'))
                                        if pd.isna(v1) or pd.isna(v2):
                                            continue
                                        linear_val = Real(f'neg2_{idx}_lin')
                                        opt2.add(linear_val == coeff1_2 * v1 + coeff2_2 * v2)
                                        neg_soft_constraints_2.append(linear_val > threshold_2)
                                    
                                    if pos_satisfied_2:
                                        # For second rule, try to maximize precision (minimize false positives)
                                        # This helps find a complementary constraint
                                        # Count negatives that DON'T satisfy (we want to minimize these)
                                        neg_not_satisfied = []
                                        for idx in neg_indices:
                                            v1 = float(pd.to_numeric(X.loc[idx, col1], errors='coerce'))
                                            v2 = float(pd.to_numeric(X.loc[idx, col2], errors='coerce'))
                                            if pd.isna(v1) or pd.isna(v2):
                                                continue
                                            linear_val = Real(f'neg2_ns_{idx}_lin')
                                            opt2.add(linear_val == coeff1_2 * v1 + coeff2_2 * v2)
                                            # Negatives should NOT satisfy (linear_val > threshold_2)
                                            neg_not_satisfied.append(If(linear_val > threshold_2, 1, 0))
                                        
                                        # Maximize positives that satisfy AND negatives that don't satisfy
                                        # This maximizes precision
                                        if neg_not_satisfied:
                                            opt2.maximize(Sum(pos_satisfied_2) + Sum(neg_not_satisfied))
                                        else:
                                            opt2.maximize(Sum(pos_satisfied_2))
                                        
                                        for neg_constraint in neg_soft_constraints_2:
                                            opt2.add_soft(neg_constraint, weight=1)
                                        
                                        if opt2.check() == sat:
                                            model2 = opt2.model()
                                            coeff1_val_2 = float(model2[coeff1_2].as_fraction())
                                            coeff2_val_2 = float(model2[coeff2_2].as_fraction())
                                            threshold_val_2 = float(model2[threshold_2].as_fraction())
                                            
                                            # Evaluate second rule
                                            linear_comb_2 = coeff1_val_2 * val1_series + coeff2_val_2 * val2_series
                                            mask_2 = linear_comb_2 <= threshold_val_2
                                            
                                            pos_covered_2 = mask_2[y_series == 1].sum()
                                            neg_covered_2 = mask_2[y_series == 0].sum()
                                            total_pos = (y_series == 1).sum()
                                            total_neg = (y_series == 0).sum()
                                            
                                            pos_coverage_2 = pos_covered_2 / max(total_pos, 1)
                                            neg_exclusion_2 = 1 - (neg_covered_2 / max(total_neg, 1))
                                            precision_2 = pos_covered_2 / max(mask_2.sum(), 1) if mask_2.sum() > 0 else 0
                                            recall_2 = pos_coverage_2
                                            f1_2 = 2 * precision_2 * recall_2 / (precision_2 + recall_2) if (precision_2 + recall_2) > 0 else 0
                                            
                                            if precision_2 > 0.5 and pos_coverage_2 > 0.2:
                                                arithmetic_rule_2 = {
                                                    'type': 'arithmetic_linear',
                                                    'features': [col1, col2],
                                                    'coefficients': [coeff1_val_2, coeff2_val_2],
                                                    'threshold': threshold_val_2,
                                                    'operation': '<=',
                                                    'pos_coverage': pos_coverage_2,
                                                    'neg_exclusion': neg_exclusion_2,
                                                    'precision': precision_2,
                                                    'recall': recall_2,
                                                    'f1': f1_2,
                                                    'support': mask_2.sum() / len(X),
                                                    'score': 0.4 * pos_coverage_2 + 0.3 * neg_exclusion_2 + 0.2 * precision_2,
                                                    'verified': True,
                                                    'smt_optimized': True
                                                }
                                                arithmetic_rules.append(arithmetic_rule_2)
                                                rules_found_for_pair += 1
                                                
                                                if self.verbose:
                                                    print(f"  [OK] Learned (2nd): {coeff1_val_2:.4f}*{col1} + {coeff2_val_2:.4f}*{col2} <= {threshold_val_2:.4f} "
                                                          f"(precision={precision_2:.2f}, coverage={pos_coverage_2:.2f})")
                                                
                                                # Update covered positives
                                                current_covered_2 = set(idx for idx in pos_indices if mask_2.loc[idx])
                                                covered_positives.update(current_covered_2)
                except Exception as e:
                            if self.verbose:
                                print(
    f"  Error learning arithmetic for {col1}, {col2}: {e}")
                            continue
        
        # Try 3-variable combinations for 3D problems
        if len(numeric_cols) >= 3:
            if self.verbose:
                print(f"[Arithmetic Learning] Also trying 3-variable combinations for 3D problems...")
            
            for i, col1 in enumerate(numeric_cols):
                for j, col2 in enumerate(numeric_cols):
                    for k, col3 in enumerate(numeric_cols):
                        if i >= j or j >= k:  # Avoid duplicates and permutations
                            continue
                        
                        # Use Z3 to find optimal coefficients AND threshold for 3 variables
                        opt = Optimize()
                        coeff1 = Real('coeff1')
                        coeff2 = Real('coeff2')
                        coeff3 = Real('coeff3')
                        threshold = Real('threshold')
                        
                        # Constrain coefficients
                        dataset_config = getattr(self, 'dataset_config', None)
                        if dataset_config and dataset_config.get('arithmetic_bounds'):
                            MIN_COEFF, MAX_COEFF = dataset_config['arithmetic_bounds']
                        else:
                            target_pred = getattr(self, 'target_predicate', None)
                            if target_pred in ['halfplane', 'interval', 'target', 'halfplane3d']:
                                MIN_COEFF, MAX_COEFF = -100.0, 100.0
                            else:
                                MIN_COEFF, MAX_COEFF = -10.0, 10.0
                        
                        opt.add(coeff1 >= MIN_COEFF, coeff1 <= MAX_COEFF)
                        opt.add(coeff2 >= MIN_COEFF, coeff2 <= MAX_COEFF)
                        opt.add(coeff3 >= MIN_COEFF, coeff3 <= MAX_COEFF)
                        opt.add(Or(coeff1 != 0, coeff2 != 0, coeff3 != 0))
                        
                        try:
                            val1_series = pd.to_numeric(X[col1], errors='coerce').astype(float)
                            val2_series = pd.to_numeric(X[col2], errors='coerce').astype(float)
                            val3_series = pd.to_numeric(X[col3], errors='coerce').astype(float)
                            
                            if val1_series.isna().all() or val2_series.isna().all() or val3_series.isna().all():
                                continue
                            
                            val1_min, val1_max = float(val1_series.min()), float(val1_series.max())
                            val2_min, val2_max = float(val2_series.min()), float(val2_series.max())
                            val3_min, val3_max = float(val3_series.min()), float(val3_series.max())
                            
                            max_linear = MAX_COEFF * max(abs(val1_min), abs(val1_max)) + \
                                        MAX_COEFF * max(abs(val2_min), abs(val2_max)) + \
                                        MAX_COEFF * max(abs(val3_min), abs(val3_max))
                            opt.add(threshold >= -max_linear)
                            opt.add(threshold <= max_linear)
                            
                            pos_satisfied = []
                            neg_soft_constraints = []
                            
                            for idx in pos_indices:
                                v1 = float(pd.to_numeric(X.loc[idx, col1], errors='coerce'))
                                v2 = float(pd.to_numeric(X.loc[idx, col2], errors='coerce'))
                                v3 = float(pd.to_numeric(X.loc[idx, col3], errors='coerce'))
                                if pd.isna(v1) or pd.isna(v2) or pd.isna(v3):
                                    continue
                                
                                linear_val = Real(f'pos_{idx}_lin3')
                                opt.add(linear_val == coeff1 * v1 + coeff2 * v2 + coeff3 * v3)
                                pos_satisfied.append(If(linear_val <= threshold, 1, 0))
                            
                            for idx in neg_indices:
                                v1 = float(pd.to_numeric(X.loc[idx, col1], errors='coerce'))
                                v2 = float(pd.to_numeric(X.loc[idx, col2], errors='coerce'))
                                v3 = float(pd.to_numeric(X.loc[idx, col3], errors='coerce'))
                                if pd.isna(v1) or pd.isna(v2) or pd.isna(v3):
                                    continue
                                
                                linear_val = Real(f'neg_{idx}_lin3')
                                opt.add(linear_val == coeff1 * v1 + coeff2 * v2 + coeff3 * v3)
                                neg_soft_constraints.append(linear_val > threshold)
                            
                            if not pos_satisfied:
                                continue
                            
                            opt.maximize(Sum(pos_satisfied))
                            for neg_constraint in neg_soft_constraints:
                                opt.add_soft(neg_constraint, weight=1)
                            
                            if opt.check() == sat:
                                model = opt.model()
                                coeff1_val = float(model[coeff1].as_fraction())
                                coeff2_val = float(model[coeff2].as_fraction())
                                coeff3_val = float(model[coeff3].as_fraction())
                                threshold_val = float(model[threshold].as_fraction())
                                
                                # REFINEMENT: Iteratively refine coefficients to get closer to true values
                                # Run 2-3 refinement iterations with tighter bounds around current solution
                                best_coeffs = [coeff1_val, coeff2_val, coeff3_val]
                                best_threshold = threshold_val
                                best_precision = 0
                                best_pos_coverage = 0
                                
                                for refine_iter in range(5):  # 5 refinement iterations for better convergence
                                    # Create new optimizer with tighter bounds around current solution
                                    refine_opt = Optimize()
                                    refine_coeff1 = Real('refine_coeff1')
                                    refine_coeff2 = Real('refine_coeff2')
                                    refine_coeff3 = Real('refine_coeff3')
                                    refine_threshold = Real('refine_threshold')
                                    
                                    # Tighter bounds: ±20% of current values, but still within original bounds
                                    refine_range = 0.2  # 20% refinement range
                                    refine_opt.add(refine_coeff1 >= max(MIN_COEFF, best_coeffs[0] * (1 - refine_range)))
                                    refine_opt.add(refine_coeff1 <= min(MAX_COEFF, best_coeffs[0] * (1 + refine_range)))
                                    refine_opt.add(refine_coeff2 >= max(MIN_COEFF, best_coeffs[1] * (1 - refine_range)))
                                    refine_opt.add(refine_coeff2 <= min(MAX_COEFF, best_coeffs[1] * (1 + refine_range)))
                                    refine_opt.add(refine_coeff3 >= max(MIN_COEFF, best_coeffs[2] * (1 - refine_range)))
                                    refine_opt.add(refine_coeff3 <= min(MAX_COEFF, best_coeffs[2] * (1 + refine_range)))
                                    refine_opt.add(refine_threshold >= best_threshold * (1 - refine_range))
                                    refine_opt.add(refine_threshold <= best_threshold * (1 + refine_range))
                                    
                                    # Rebuild constraints with refined variables
                                    refine_pos_satisfied = []
                                    refine_neg_constraints = []
                                    
                                    for idx in pos_indices:
                                        v1 = float(pd.to_numeric(X.loc[idx, col1], errors='coerce'))
                                        v2 = float(pd.to_numeric(X.loc[idx, col2], errors='coerce'))
                                        v3 = float(pd.to_numeric(X.loc[idx, col3], errors='coerce'))
                                        if pd.isna(v1) or pd.isna(v2) or pd.isna(v3):
                                            continue
                                        
                                        linear_val = Real(f'refine_pos_{idx}_lin3')
                                        refine_opt.add(linear_val == refine_coeff1 * v1 + refine_coeff2 * v2 + refine_coeff3 * v3)
                                        refine_pos_satisfied.append(If(linear_val <= refine_threshold, 1, 0))
                                    
                                    for idx in neg_indices:
                                        v1 = float(pd.to_numeric(X.loc[idx, col1], errors='coerce'))
                                        v2 = float(pd.to_numeric(X.loc[idx, col2], errors='coerce'))
                                        v3 = float(pd.to_numeric(X.loc[idx, col3], errors='coerce'))
                                        if pd.isna(v1) or pd.isna(v2) or pd.isna(v3):
                                            continue
                                        
                                        linear_val = Real(f'refine_neg_{idx}_lin3')
                                        refine_opt.add(linear_val == refine_coeff1 * v1 + refine_coeff2 * v2 + refine_coeff3 * v3)
                                        refine_neg_constraints.append(linear_val > refine_threshold)
                                    
                                    if refine_pos_satisfied:
                                        refine_opt.maximize(Sum(refine_pos_satisfied))
                                        for neg_constraint in refine_neg_constraints:
                                            refine_opt.add_soft(neg_constraint, weight=1)
                                        
                                        if refine_opt.check() == sat:
                                            refine_model = refine_opt.model()
                                            refine_coeff1_val = float(refine_model[refine_coeff1].as_fraction())
                                            refine_coeff2_val = float(refine_model[refine_coeff2].as_fraction())
                                            refine_coeff3_val = float(refine_model[refine_coeff3].as_fraction())
                                            refine_threshold_val = float(refine_model[refine_threshold].as_fraction())
                                            
                                            # Evaluate refined rule
                                            refine_linear_comb = refine_coeff1_val * val1_series + refine_coeff2_val * val2_series + refine_coeff3_val * val3_series
                                            refine_mask = refine_linear_comb <= refine_threshold_val
                                            
                                            refine_pos_covered = refine_mask[y_series == 1].sum()
                                            refine_neg_covered = refine_mask[y_series == 0].sum()
                                            refine_total_pos = (y_series == 1).sum()
                                            
                                            refine_pos_coverage = refine_pos_covered / max(refine_total_pos, 1)
                                            refine_precision = refine_pos_covered / max(refine_mask.sum(), 1) if refine_mask.sum() > 0 else 0
                                            
                                            # Keep if better
                                            if refine_precision > best_precision or (refine_precision == best_precision and refine_pos_coverage > best_pos_coverage):
                                                best_coeffs = [refine_coeff1_val, refine_coeff2_val, refine_coeff3_val]
                                                best_threshold = refine_threshold_val
                                                best_precision = refine_precision
                                                best_pos_coverage = refine_pos_coverage
                                
                                # Use best coefficients from refinement
                                coeff1_val, coeff2_val, coeff3_val = best_coeffs
                                threshold_val = best_threshold
                                
                                linear_comb_series = coeff1_val * val1_series + coeff2_val * val2_series + coeff3_val * val3_series
                                mask = linear_comb_series <= threshold_val
                                
                                pos_covered = mask[y_series == 1].sum()
                                neg_covered = mask[y_series == 0].sum()
                                total_pos = (y_series == 1).sum()
                                total_neg = (y_series == 0).sum()
                                
                                pos_coverage = pos_covered / max(total_pos, 1)
                                neg_exclusion = 1 - (neg_covered / max(total_neg, 1))
                                precision = pos_covered / max(mask.sum(), 1) if mask.sum() > 0 else 0
                                recall = pos_coverage
                                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                                
                                if precision > 0.5 and pos_coverage > 0.3:
                                    arithmetic_rule = {
                                        'type': 'arithmetic_linear',
                                        'features': [col1, col2, col3],
                                        'coefficients': [coeff1_val, coeff2_val, coeff3_val],
                                        'threshold': threshold_val,
                                        'operation': '<=',
                                        'pos_coverage': pos_coverage,
                                        'neg_exclusion': neg_exclusion,
                                        'precision': precision,
                                        'recall': recall,
                                        'f1': f1,
                                        'support': mask.sum() / len(X),
                                        'score': 0.4 * pos_coverage + 0.3 * neg_exclusion + 0.2 * precision,
                                        'verified': True,
                                        'smt_optimized': True
                                    }
                                    arithmetic_rules.append(arithmetic_rule)
                                    
                                    if self.verbose:
                                        print(f"  [OK] Learned (3D, refined): {coeff1_val:.4f}*{col1} + {coeff2_val:.4f}*{col2} + {coeff3_val:.4f}*{col3} <= {threshold_val:.4f} "
                                              f"(precision={precision:.2f}, coverage={pos_coverage:.2f})")
                        except Exception as e:
                            if self.verbose:
                                print(f"  Error learning 3-variable arithmetic for {col1}, {col2}, {col3}: {e}")
                            continue

        if self.verbose:
            print(f"[Arithmetic Learning] Learned {len(arithmetic_rules)} arithmetic relationships")

        return arithmetic_rules

    def _learn_range_relationships_smt(
    self,
    rules: List[Dict],
    X: pd.DataFrame,
     y: np.ndarray) -> List[Dict]:
        """
        Learn range relationships (lower < X < upper) for interval problems

        This method uses SMT to learn both lower and upper bounds simultaneously.
        For interval problems, we need rules like: lower < X < upper
        """
        if not self.use_optimization:
            return []

        range_rules = []
        y_series = pd.Series(y, index=X.index)
        pos_indices = X.index[y_series == 1].tolist()
        neg_indices = X.index[y_series == 0].tolist()

        if len(pos_indices) == 0 or len(neg_indices) == 0:
            return []

        # Get numeric columns (for interval, typically just one: 'x')
        numeric_cols = [
    col for col in X.columns if pd.api.types.is_numeric_dtype(
        X[col])]
        if len(numeric_cols) == 0:
            return []

        if self.verbose:
            print(f"\n[Range Learning] Attempting to learn range relationships from {len(numeric_cols)} numeric features...")

        # Learn range: lower < feature < upper
        for col in numeric_cols:
            try:
                val_series = pd.to_numeric(
    X[col], errors='coerce').astype(float)
                if val_series.isna().all():
                    continue

                # Use Z3 to find optimal lower and upper bounds simultaneously
                opt = Optimize()
                lower_bound = Real('lower_bound')
                upper_bound = Real('upper_bound')

                # Get feature value ranges
                feat_min = float(val_series.min())
                feat_max = float(val_series.max())

                # Constrain bounds
                opt.add(lower_bound >= feat_min)
                opt.add(lower_bound <= feat_max)
                opt.add(upper_bound >= feat_min)
                opt.add(upper_bound <= feat_max)
                # Lower must be less than upper
                opt.add(lower_bound < upper_bound)

                # Create constraints: positives should satisfy, negatives
                # should not
                pos_satisfied = []
                neg_soft_constraints = []

                for idx in pos_indices:
                    try:
                        v_val = X.loc[idx, col]
                        if pd.isna(v_val):
                            continue
                        v = float(pd.to_numeric(v_val, errors='coerce'))
                        if pd.isna(v):
                            continue

                        # Positive: lower < v < upper
                        pos_satisfied.append(
                            If(And(lower_bound < v, v < upper_bound), 1, 0))
                    except (ValueError, TypeError):
                        continue

                for idx in neg_indices:
                    try:
                        v_val = X.loc[idx, col]
                        if pd.isna(v_val):
                            continue
                        v = float(pd.to_numeric(v_val, errors='coerce'))
                        if pd.isna(v):
                            continue

                        # Negative: prefer v < lower OR v > upper
                        neg_soft_constraints.append(
                            Or(v < lower_bound, v > upper_bound))
                    except (ValueError, TypeError):
                        continue

                if not pos_satisfied:
                    continue

                # Maximize positive coverage
                opt.maximize(Sum(pos_satisfied))

                # Soft constraints: prefer negatives don't satisfy
                for neg_constraint in neg_soft_constraints:
                    opt.add_soft(neg_constraint, weight=1)

                # Solve
                if opt.check() == sat:
                    model = opt.model()
                    lower_val = float(model[lower_bound].as_fraction())
                    upper_val = float(model[upper_bound].as_fraction())

                    # Evaluate the rule
                    mask = (val_series > lower_val) & (val_series < upper_val)

                    pos_covered = mask[y_series == 1].sum()
                    neg_covered = mask[y_series == 0].sum()
                    total_pos = (y_series == 1).sum()
                    total_neg = (y_series == 0).sum()

                    pos_coverage = pos_covered / max(total_pos, 1)
                    neg_exclusion = 1 - (neg_covered / max(total_neg, 1))
                    precision = pos_covered / \
                        max(mask.sum(), 1) if mask.sum() > 0 else 0

                    # Only keep if it's better than random (precision > 0.5)
                    if precision > 0.5 and pos_coverage > 0.3:
                        recall = pos_coverage  # For range rules, recall = pos_coverage
                        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                        range_rule = {
                            'type': 'range',
                            'feature': col,
                            'lower_bound': lower_val,
                            'upper_bound': upper_val,
                            'pos_coverage': pos_coverage,
                            'neg_exclusion': neg_exclusion,
                            'precision': precision,
                            'recall': recall,
                            'f1': f1,
                            'support': mask.sum() / len(X),
                            'score': 0.4 * pos_coverage + 0.3 * neg_exclusion + 0.2 * precision,
                            'verified': True,
                            'smt_optimized': True
                        }
                        range_rules.append(range_rule)

                        if self.verbose:
                            print(f"  [OK] Learned: {lower_val:.4f} < {col} < {upper_val:.4f} "
                                  f"(precision={precision:.2f}, coverage={pos_coverage:.2f})")
            except Exception as e:
                if self.verbose:
                    print(f"  Error learning range for {col}: {e}")
                continue

        if self.verbose:
            print(f"[Range Learning] Learned {len(range_rules)} range relationships")

        return range_rules

    def _verify_rules_with_smt(
    self,
    rules: List[Dict],
    X: pd.DataFrame,
     y: np.ndarray) -> List[Dict]:
        """
        Phase 1: Enhanced SMT fitting, instantiation, and scoring
        
        Uses Optimize() for threshold optimization with soft constraints for negative examples and blocking constraints for UNSAT rules
        Performs SMT fitting per clause: declare unknown thresholds as SMT variables and assert training-wide constraints
        Instantiation and filtering: if satisfiable, substitute constants from model
        - Otherwise discard or add blocking constraints
        
        Scoring and selection: combine coverage/precision/recall/support and apply compression/utility preferences
        """
        if self.verbose:
            print(f"\n[SMT Fitting] Optimizing {len(rules)} rules with SMT...")
        
        verified = []
        y_series = pd.Series(y, index=X.index)
        pos_indices = X.index[y_series == 1].tolist()
        neg_indices = X.index[y_series == 0].tolist()
        
        for rule_idx, rule in enumerate(rules):
            try:
                # Numsynth-style rules are already optimized, skip SMT optimization
                if rule.get('numsynth_style', False) and rule.get('smt_optimized', False):
                    # These rules were generated with SMT optimization already
                    # Just verify they're satisfiable and calculate metrics
                    pass
                
                # Range rules are already optimized by SMT, just verify and preserve metrics
                if rule.get('type') == 'range' and rule.get('smt_optimized', False):
                    # Range rules are already optimized, just verify they work
                    try:
                        mask = self._apply_rule(rule, X)
                        pos_covered = mask[y_series == 1].sum()
                        neg_covered = mask[y_series == 0].sum()
                        total_pos = (y_series == 1).sum()
                        total_neg = (y_series == 0).sum()
                        
                        # Preserve original recall and f1 if they exist, otherwise recalculate
                        if 'recall' not in rule or 'f1' not in rule:
                            pos_coverage = pos_covered / max(total_pos, 1)
                            precision = pos_covered / max(mask.sum(), 1) if mask.sum() > 0 else 0
                            recall = pos_coverage
                            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                            rule['recall'] = recall
                            rule['f1'] = f1
                            rule['precision'] = precision
                            rule['pos_coverage'] = pos_coverage
                        
                        verified.append(rule)
                    except Exception as e:
                        if self.verbose:
                            print(f"  Error verifying range rule {rule_idx}: {e}")
                        continue
                    continue  # Skip to next rule
                
                # SMT Fitting - optimize thresholds
                if rule['type'] == 'single_feature':
                    feature = rule['feature']
                    if feature not in X.columns:
                        continue
                    
                    operation = rule['operation']
                    if operation in ['>', '<'] and rule.get(
                        'has_optimizable_threshold'):
                        # Optimize threshold using SMT
                        optimized_threshold = self._optimize_threshold_smt(
                            rule, X, y, feature, operation, rule['threshold']
                        )
                        # Ensure optimized threshold is always float
                        if optimized_threshold is not None:
                            rule['threshold'] = float(optimized_threshold)
                        rule['threshold_optimized'] = True
                    elif operation == '==':
                        # Equality: For geometry problems, == with threshold 0 is too restrictive
                        # Convert to > or < if threshold is 0 or very close to
                        # 0
                        threshold_val = rule.get('threshold', 0)
                        if abs(threshold_val) < 1e-6:  # Threshold is essentially 0
                            # Convert == 0 to a more general inequality
                            # For geometry, prefer < 0 or > 0 based on data
                            # distribution
                            pos_vals = X.loc[pos_indices, feature] if len(
                                pos_indices) > 0 else pd.Series()
                            neg_vals = X.loc[neg_indices, feature] if len(
                                neg_indices) > 0 else pd.Series()

                            if len(pos_vals) > 0 and len(neg_vals) > 0:
                                # Check if positives tend to be < 0 or > 0
                                pos_mean = float(pd.to_numeric(
                                    pos_vals, errors='coerce').mean())
                                neg_mean = float(pd.to_numeric(
                                    neg_vals, errors='coerce').mean())

                                # Convert to inequality that better separates
                                # positives from negatives
                                if pos_mean < neg_mean:
                                    # Positives are smaller, use < threshold
                                    rule['operation'] = '<'
                                    # Optimize the threshold
                                    optimized_threshold = self._optimize_threshold_smt(
                                        rule, X, y, feature, '<', threshold_val
                                    )
                                    if optimized_threshold is not None:
                                        rule['threshold'] = float(
                                            optimized_threshold)
                                else:
                                    # Positives are larger, use > threshold
                                    rule['operation'] = '>'
                                    # Optimize the threshold
                                    optimized_threshold = self._optimize_threshold_smt(
                                        rule, X, y, feature, '>', threshold_val
                                    )
                                    if optimized_threshold is not None:
                                        rule['threshold'] = float(
                                            optimized_threshold)
                                rule['threshold_optimized'] = True
                            else:
                                rule['threshold_optimized'] = False
                        else:
                            # Non-zero equality: keep as is (too specific,
                            # might be intentional)
                            rule['threshold_optimized'] = False
                    elif operation == 'in':
                        # Optimize categorical set membership
                        optimized_set = self._optimize_categorical_smt(
                            rule, X, y, feature, rule.get('value', []))
                        if optimized_set:
                            rule['value'] = optimized_set
                            rule['threshold_optimized'] = True
                        else:
                            rule['threshold_optimized'] = False
                
                elif rule['type'] == 'conjunction':
                    # Joint threshold optimization for conjunctions
                    # Optimize all thresholds together rather than separately
                    optimized_components = self._optimize_conjunction_jointly(
                        rule, X, y
                    )
                    if optimized_components:
                        rule['components'] = optimized_components
                        rule['jointly_optimized'] = True
                    else:
                        # Fallback: optimize each component separately
                        for comp_idx, comp in enumerate(
                            rule.get('components', [])):
                            if comp.get('operation') in ['>', '<']:
                                feature = comp['feature']
                                if feature in X.columns:
                                    optimized_threshold = self._optimize_threshold_smt(
                                        rule, X, y, feature, comp['operation'], comp['threshold']
                                    )
                                    # Ensure optimized threshold is always float
                                    if optimized_threshold is not None:
                                        comp['threshold'] = float(
                                            optimized_threshold)
                                    comp['threshold_optimized'] = True
                
                # Instantiation & filtering - verify rule is satisfiable
                # Check if rule can be satisfied with optimized thresholds
                solver = Solver()
                satisfiable = True
                
                if rule['type'] == 'single_feature':
                    feature = rule['feature']
                    operation = rule['operation']
                    
                    if operation in ['>', '<']:
                        threshold_var = Real('t')
                        solver.add(threshold_var == rule['threshold'])
                        # Check if there exists a value that satisfies
                        feat_var = Real('f')
                        if operation == '>':
                            solver.add(feat_var > threshold_var)
                        else:
                            solver.add(feat_var < threshold_var)
                        
                        result = solver.check()
                        satisfiable = (result == sat)
                    elif operation == '==':
                        satisfiable = True  # Equality is always satisfiable
                
                elif rule['type'] == 'conjunction':
                    # Check if all components can be satisfied together
                    constraints = []
                    for comp in rule.get('components', []):
                        if comp.get('operation') in ['>', '<']:
                            threshold_var = Real(f"t_{comp['feature']}")
                            solver.add(threshold_var == comp['threshold'])
                            feat_var = Real(f"f_{comp['feature']}")
                            if comp['operation'] == '>':
                                constraints.append(feat_var > threshold_var)
                            else:
                                constraints.append(feat_var < threshold_var)
                    
                    if constraints:
                        solver.add(And(*constraints))
                        result = solver.check()
                        satisfiable = (result == sat)
                
                if not satisfiable:
                    # Store blocking constraint to prevent similar
                    # bad rules
                    self._add_blocking_constraint(rule)
                    if self.verbose:
                        print(
    f"  Rule {rule_idx} unsatisfiable after optimization, discarding and blocking similar rules")
                    continue
                
                # Scoring & selection
                try:
                    mask = self._apply_rule(rule, X)
                except Exception as e:
                    if self.verbose:
                        print(f"Error in _apply_rule for rule {rule_idx}: {e}")
                        print(f"Rule: {rule}")
                        import traceback
                        traceback.print_exc()
                    # Skip this rule
                    continue

                # Post-process rules to add categorical constraints
                # If rule has categorical variables (from PyGol), add constraints based on training data
                # This converts rules like color(A,B) to color(A,'red') based on what positives match
                # Check all rules for categorical variables, not just those with the flag
                # The flag might be lost during filtering, so we need to check the actual components
                if self.categorical_features:
                    # Check if rule has any categorical features that might be variables
                    has_categorical = False
                    if rule.get('type') == 'single_feature':
                        if rule.get('feature') in self.categorical_features:
                            has_categorical = True
                    elif rule.get('type') == 'conjunction':
                        for comp in rule.get('components', []):
                            if comp.get('feature') in self.categorical_features:
                                has_categorical = True
                                break
                    
                    # If rule has categorical features, ensure they're not variables
                    if has_categorical:
                        rule = self._add_categorical_constraints(rule, X, y, mask)
                        # Verify constraints were applied
                        if self.verbose and rule.get('type') == 'conjunction':
                            for comp in rule.get('components', []):
                                if comp.get('operation') == 'variable' and comp.get('feature') in (self.categorical_features or []):
                                    print(f"  ERROR: Categorical variable {comp.get('feature')} still present after _add_categorical_constraints!")
                                    print(f"    Component: {comp}")
                                    print(f"    Rule source: {rule.get('pygol_source', 'unknown')}")

                pos_covered = mask[y_series == 1].sum()
                neg_covered = mask[y_series == 0].sum()
                total_pos = (y_series == 1).sum()
                total_neg = (y_series == 0).sum()
                
                pos_coverage = pos_covered / max(total_pos, 1)
                neg_exclusion = 1 - (neg_covered / max(total_neg, 1))
                support = mask.sum() / len(X)
                
                # Recalculate mask after adding categorical constraints
                try:
                    mask = self._apply_rule(rule, X)
                    pos_covered = mask[y_series == 1].sum()
                    neg_covered = mask[y_series == 0].sum()
                    pos_coverage = pos_covered / max(total_pos, 1)
                    neg_exclusion = 1 - (neg_covered / max(total_neg, 1))
                    support = mask.sum() / len(X)
                except:
                    pass  # If recalculation fails, use original metrics
                
                precision = pos_covered / \
                    max(mask.sum(), 1) if mask.sum() > 0 else 0
                recall = pos_coverage
                f1 = 2 * precision * recall / \
                    (precision + recall) if (precision + recall) > 0 else 0
                
                num_literals = 1 if rule['type'] == 'single_feature' else len(rule.get('components', []))
                compression_score = 1.0 / (1.0 + num_literals * 0.1)
                accuracy_weight = 0.5 * precision + 0.3 * recall + 0.2 * f1
                
                rule['objectives'] = {
                    'pos_coverage': pos_coverage,
                    'neg_exclusion': neg_exclusion,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'support': support,
                    'compression': compression_score
                }
                
                rule['score'] = (0.3 * pos_coverage + 0.2 * neg_exclusion + 0.4 * precision + 0.1 * compression_score)
                rule['precision'] = precision
                rule['recall'] = recall
                rule['f1'] = f1
                rule['support'] = support
                rule['pos_coverage'] = pos_coverage
                rule['neg_exclusion'] = neg_exclusion
                rule['compression'] = compression_score
                rule['num_literals'] = num_literals
                rule['verified'] = True
                rule['smt_optimized'] = True
                verified.append(rule)
                
            except Exception as e:
                if self.verbose:
                    print(f"Error in SMT fitting for rule {rule_idx}: {e}")
                continue
        
        if self.verbose:
            print(f"[OK] Optimized and verified {len(verified)} rules (discarded {len(rules) - len(verified)})")
        
        return verified
    
    def _add_categorical_constraints(self, rule: Dict, X: pd.DataFrame, y: np.ndarray, mask: pd.Series) -> Dict:
        """Post-process rules to add categorical constraints based on training data."""
        if not hasattr(self, 'categorical_features') or not self.categorical_features:
            return rule
        
        pos_mask = mask & (pd.Series(y, index=X.index) == 1)
        pos_indices = X.index[pos_mask].tolist()
        
        if not pos_indices:
            return rule
        # Look for categorical features in rule components that don't have specific values
        rule_modified = False
        
        if rule['type'] == 'single_feature':
            feature = rule.get('feature', '')
            if feature in self.categorical_features and rule.get('operation') != '==':
                # This is a categorical feature used as a variable - optimize using SMT
                # Same approach as threshold optimization: PyGol learns structure, SMT optimizes value
                optimized_value = self._optimize_categorical_value_smt(
                    feature, pos_indices, X, y
                )
                if optimized_value is not None:
                    most_common = optimized_value
                else:
                    # Fallback: use most common value if SMT optimization fails
                    pos_values = X.loc[pos_indices, feature].value_counts()
                    if len(pos_values) > 0:
                        most_common = pos_values.index[0]
                        # Decode if needed
                        if feature in self.feature_encoders:
                            try:
                                if pd.api.types.is_numeric_dtype(type(most_common)) or isinstance(most_common, (int, float)):
                                    value_int = int(float(most_common))
                                    encoder = self.feature_encoders[feature]
                                    if hasattr(encoder, 'classes_') and value_int < len(encoder.classes_):
                                        most_common = encoder.classes_[value_int]
                            except:
                                pass
                    else:
                        return rule  # Can't optimize
                
                # Convert to equality constraint
                rule['operation'] = '=='
                rule['value'] = most_common
                rule_modified = True
                if self.verbose:
                    print(f"  [Categorical Optimization] Optimized {feature}(X, ?) → {feature}(X, '{most_common}') using SMT")
        
        elif rule['type'] == 'conjunction':
            components = rule.get('components', [])
            for comp_idx, comp in enumerate(components):
                feature = comp.get('feature', '')
                # Check if this is a categorical variable that needs to be replaced
                if feature in self.categorical_features and comp.get('operation') == 'variable':
                    # This is a categorical feature used as a variable - need to constrain it
                    # Find most common categorical value in positive examples that match other components
                    # First, apply other components to filter examples
                    other_mask = pd.Series(True, index=X.index)
                    for other_comp in components:
                        if other_comp != comp and other_comp.get('operation') != 'variable':
                            try:
                                other_mask = other_mask & self._apply_component(other_comp, X)
                            except:
                                pass
                    
                    # Find positives that match other components
                    filtered_pos = X.index[(other_mask & (pd.Series(y, index=X.index) == 1))].tolist()
                    
                    # If filtered_pos is empty, use all positives
                    # fall back to ALL positive examples to find the most common value
                    # This ensures we always replace the variable, even if the rule is very specific
                    if not filtered_pos:
                        filtered_pos = X.index[(pd.Series(y, index=X.index) == 1)].tolist()
                        if self.verbose:
                            print(f"  [Categorical Constraint] No positives match other components for {feature}, using all positives")
                    
                    if filtered_pos:
                        pos_values = X.loc[filtered_pos, feature].value_counts()
                        if len(pos_values) > 0:
                            most_common = pos_values.index[0]
                            # Decode if needed
                            if feature in self.feature_encoders:
                                try:
                                    if pd.api.types.is_numeric_dtype(type(most_common)) or isinstance(most_common, (int, float)):
                                        value_int = int(float(most_common))
                                        encoder = self.feature_encoders[feature]
                                        if hasattr(encoder, 'classes_') and value_int < len(encoder.classes_):
                                            most_common = encoder.classes_[value_int]
                                except:
                                    pass
                            
                            # Use SMT to optimize categorical value selection
                            # Instead of just using most common, find the value that maximizes precision
                            optimized_value = self._optimize_categorical_value_smt(
                                feature, filtered_pos, X, y
                            )
                            if optimized_value is not None:
                                most_common = optimized_value
                            
                            # Convert to equality constraint
                            comp['operation'] = '=='
                            comp['value'] = most_common
                            # Remove the variable marker
                            if 'variable_name' in comp:
                                del comp['variable_name']
                            rule_modified = True
                            if self.verbose:
                                print(f"  [Categorical Constraint] Replaced {feature}(X, ?) with {feature}(X, '{most_common}')")
                                # Verify the replacement worked
                                if comp.get('operation') != '==' or comp.get('value') != most_common:
                                    print(f"  [Categorical Constraint] ERROR: Replacement failed! operation={comp.get('operation')}, value={comp.get('value')}")
                        else:
                            if self.verbose:
                                print(f"  [Categorical Constraint] WARNING: No values found for {feature}, cannot replace variable")
                    else:
                        if self.verbose:
                            print(f"  [Categorical Constraint] WARNING: No positive examples found for {feature}, cannot replace variable")
        
        if rule_modified and self.verbose:
            print(f"  [Categorical Constraint] Added categorical constraints to rule")
        
        return rule
    
    def _apply_component(self, comp: Dict, X: pd.DataFrame) -> pd.Series:
        """Apply a single rule component to data"""
        feature = comp.get('feature', '')
        operation = comp.get('operation', '>')
        
        if feature not in X.columns:
            return pd.Series(False, index=X.index)
        
        try:
            if operation == '>':
                threshold = comp.get('threshold', 0)
                return pd.to_numeric(X[feature], errors='coerce') > threshold
            elif operation == '<':
                threshold = comp.get('threshold', 0)
                return pd.to_numeric(X[feature], errors='coerce') < threshold
            elif operation == '==':
                value = comp.get('value', comp.get('threshold', 0))
                # Decode categorical values before comparison
                # X[feature] is encoded as a number (0, 1, 2), but value is a string ('red', 'blue')
                # We need to encode value to match X[feature], or decode X[feature] to match value
                if feature in (self.categorical_features or []):
                    # This is a categorical feature - value is a string, X[feature] is encoded
                    # Encode value to match X[feature]
                    if feature in self.feature_encoders:
                        encoder = self.feature_encoders[feature]
                        try:
                            # Encode the string value to get the numeric encoding
                            if isinstance(value, str):
                                # Value is a string like 'red' - encode it
                                encoded_value = encoder.transform([value])[0]
                                return X[feature] == encoded_value
                            else:
                                # Value is already encoded - use directly
                                return X[feature] == value
                        except (ValueError, KeyError, IndexError):
                            # If encoding fails (value not in encoder), return False
                            return pd.Series(False, index=X.index)
                    else:
                        # No encoder - compare directly (might fail if types don't match)
                        return X[feature] == value
                else:
                    # Numeric feature - compare directly
                    return X[feature] == value
            else:
                return pd.Series(False, index=X.index)
        except:
            return pd.Series(False, index=X.index)
    
    def _optimize_conjunction_jointly(self,
    rule: Dict,
    X: pd.DataFrame,
     y: np.ndarray) -> Optional[List[Dict]]:
        """
        Joint threshold optimization for conjunctions
        Optimizes all thresholds in a conjunction together using a single Optimize() instance
        This ensures thresholds are optimized jointly rather than independently
        """
        if not self.use_optimization:
            return None
        
        components = rule.get('components', [])
        if not components:
            return None
        
        y_series = pd.Series(y, index=X.index)
        pos_indices = X.index[y_series == 1].tolist()
        neg_indices = X.index[y_series == 0].tolist()
        
        if len(pos_indices) == 0 or len(neg_indices) == 0:
            return None
        
        opt = Optimize()
        threshold_vars = {}
        optimized_components = []
        
        # Create threshold variables for each component
        for comp in components:
            if comp.get('operation') in ['>', '<']:
                feature = comp['feature']
                if feature not in X.columns:
                    continue
                
                threshold_var = Real(f'thresh_{feature}')
                threshold_vars[feature] = threshold_var
                
                # Set bounds (only for numeric features)
                # Always convert to numeric first to handle mixed types safely
                try:
                    feature_series = pd.to_numeric(X[feature], errors='coerce')
                    # Force dtype to float
                    # issues
                    feature_series = feature_series.astype(float)
                    # Check if we got any valid numeric values
                    if feature_series.isna().all():
                        # All values were non-numeric, skip this feature
                        threshold_vars[feature] = None
                        continue

                    feat_min = float(feature_series.min())
                    feat_max = float(feature_series.max())
                    opt.add(threshold_var >= feat_min)
                    opt.add(threshold_var <= feat_max)
                except (TypeError, ValueError) as e:
                    # If conversion fails, skip this feature
                    threshold_vars[feature] = None
                    continue
                
                # Apply blocking constraints
                self._apply_blocking_constraints(
    opt, feature, threshold_var, comp['operation'])
        
        if not threshold_vars:
            return None
        
        # Count positives and negatives that satisfy ALL components
        pos_satisfied_all = []
        neg_satisfied_all = []
        neg_soft_constraints = []
        
        for idx in pos_indices:
            component_satisfied = []
            for comp in components:
                if comp.get('operation') in ['>', '<']:
                    feature = comp['feature']
                    threshold_var = threshold_vars.get(feature)
                    if threshold_var is None:
                        continue
                    
                    feat_val = Real(f'pos_{idx}_{feature}')
                    # Ensure value is numeric before converting
                    val = X.loc[idx, feature]
                    # Try to convert to float, skip if it fails
                    try:
                        val_float = float(pd.to_numeric(val, errors='coerce'))
                        if pd.isna(val_float):
                            continue  # Skip NaN values (non-numeric)
                        opt.add(feat_val == val_float)
                    except (TypeError, ValueError):
                        continue  # Skip non-numeric values
                    
                    if comp['operation'] == '>':
                        component_satisfied.append(feat_val > threshold_var)
                    elif comp['operation'] == '<':
                        component_satisfied.append(feat_val < threshold_var)
                elif comp.get('operation') == '==':
                    # Equality: check directly
                    feature = comp['feature']
                    value = comp.get('value')
                    # Decode categorical values before comparison
                    if feature in (self.categorical_features or []):
                        # This is a categorical feature - value is a string, X[feature] is encoded
                        if feature in self.feature_encoders:
                            encoder = self.feature_encoders[feature]
                            try:
                                if isinstance(value, str):
                                    # Value is a string like 'red' - encode it
                                    encoded_value = encoder.transform([value])[0]
                                    if X.loc[idx, feature] == encoded_value:
                                        component_satisfied.append(True)
                                    else:
                                        component_satisfied.append(False)
                                else:
                                    # Value is already encoded - use directly
                                    if X.loc[idx, feature] == value:
                                        component_satisfied.append(True)
                                    else:
                                        component_satisfied.append(False)
                            except (ValueError, KeyError, IndexError):
                                component_satisfied.append(False)
                        else:
                            # No encoder - compare directly
                            if X.loc[idx, feature] == value:
                                component_satisfied.append(True)
                            else:
                                component_satisfied.append(False)
                    else:
                        # Numeric feature - compare directly
                        if X.loc[idx, feature] == value:
                            component_satisfied.append(True)
                        else:
                            component_satisfied.append(False)
            
            if component_satisfied:
                # All components must be satisfied (conjunction)
                all_satisfied = And(
    *
    [
        c for c in component_satisfied if isinstance(
            c,
            bool) == False]) if any(
                not isinstance(
                    c,
                    bool) for c in component_satisfied) else (
                        all(component_satisfied))
                if not isinstance(all_satisfied, bool):
                    pos_satisfied_all.append(If(all_satisfied, 1, 0))
        
        for idx in neg_indices:
            component_satisfied = []
            soft_components = []
            for comp in components:
                if comp.get('operation') in ['>', '<']:
                    feature = comp['feature']
                    threshold_var = threshold_vars.get(feature)
                    if threshold_var is None:
                        continue
                    
                    feat_val = Real(f'neg_{idx}_{feature}')
                    # Ensure value is numeric before converting
                    val = X.loc[idx, feature]
                    # Try to convert to float, skip if it fails
                    try:
                        val_float = float(pd.to_numeric(val, errors='coerce'))
                        if pd.isna(val_float):
                            continue  # Skip NaN values (non-numeric)
                        opt.add(feat_val == val_float)
                    except (TypeError, ValueError):
                        continue  # Skip non-numeric values
                    
                    if comp['operation'] == '>':
                        component_satisfied.append(feat_val > threshold_var)
                        soft_components.append(
    feat_val <= threshold_var)  # Prefer NOT satisfied
                    elif comp['operation'] == '<':
                        component_satisfied.append(feat_val < threshold_var)
                        soft_components.append(
    feat_val >= threshold_var)  # Prefer NOT satisfied
                elif comp.get('operation') == '==':
                    feature = comp['feature']
                    value = comp.get('value')
                    # Decode categorical values before comparison
                    if feature in (self.categorical_features or []):
                        # This is a categorical feature - value is a string, X[feature] is encoded
                        if feature in self.feature_encoders:
                            encoder = self.feature_encoders[feature]
                            try:
                                if isinstance(value, str):
                                    # Value is a string like 'red' - encode it
                                    encoded_value = encoder.transform([value])[0]
                                    if X.loc[idx, feature] == encoded_value:
                                        component_satisfied.append(True)
                                    else:
                                        component_satisfied.append(False)
                                else:
                                    # Value is already encoded - use directly
                                    if X.loc[idx, feature] == value:
                                        component_satisfied.append(True)
                                    else:
                                        component_satisfied.append(False)
                            except (ValueError, KeyError, IndexError):
                                component_satisfied.append(False)
                        else:
                            # No encoder - compare directly
                            if X.loc[idx, feature] == value:
                                component_satisfied.append(True)
                            else:
                                component_satisfied.append(False)
                    else:
                        # Numeric feature - compare directly
                        if X.loc[idx, feature] == value:
                            component_satisfied.append(True)
                        else:
                            component_satisfied.append(False)
            
            if component_satisfied:
                all_satisfied = And(
    *
    [
        c for c in component_satisfied if isinstance(
            c,
            bool) == False]) if any(
                not isinstance(
                    c,
                    bool) for c in component_satisfied) else (
                        all(component_satisfied))
                if not isinstance(all_satisfied, bool):
                    neg_satisfied_all.append(If(all_satisfied, 1, 0))
            
            # Soft constraints for negatives
            if soft_components:
                # Prefer that at least one component is NOT satisfied
                neg_soft_constraints.append(Or(*soft_components))
        
        # Optimize jointly
        if pos_satisfied_all:
            pos_count = Sum(pos_satisfied_all) if len(
                pos_satisfied_all) > 1 else pos_satisfied_all[0]
            opt.maximize(pos_count)
        
        if neg_satisfied_all:
            neg_count = Sum(neg_satisfied_all) if len(
                neg_satisfied_all) > 1 else neg_satisfied_all[0]
            opt.minimize(neg_count)
        
        # Add soft constraints
        for soft_constraint in neg_soft_constraints:
            opt.add_soft(soft_constraint, weight=self.soft_constraint_weight)
        
        result = opt.check()
        if result == sat:
            model = opt.model()
            # Extract optimized thresholds
            for comp in components:
                if comp.get('operation') in ['>', '<']:
                    feature = comp['feature']
                    threshold_var = threshold_vars.get(feature)
                    if threshold_var is not None:
                        try:
                            optimized = float(
    model[threshold_var].as_fraction())
                            comp['threshold'] = optimized
                            comp['threshold_optimized'] = True
                        except:
                            try:
                                optimized = float(
    model[threshold_var].as_decimal(10))
                                comp['threshold'] = optimized
                                comp['threshold_optimized'] = True
                            except:
                                pass
            return components
        else:
            return None
    
    def _optimize_categorical_value_smt(self, feature: str, filtered_pos_indices: List, 
                                        X: pd.DataFrame, y: np.ndarray) -> Optional[str]:
        """
        Optimize single categorical value using SMT (for hybrid domains)
        
        For rules with categorical variables, find the optimal categorical value that:
        1. Maximizes positive coverage (from filtered_pos_indices)
        2. Minimizes negative coverage
        
        This is similar to how numsynth optimizes categorical constraints in hybrid domains.
        Numsynth uses magic_value_type(color) and magic_value_type(orientation) to handle
        categorical values specially - it learns rules like color(X, 'red') directly.
        """
        if feature not in X.columns:
            return None
        
        y_series = pd.Series(y, index=X.index)
        pos_indices = filtered_pos_indices
        neg_indices = X.index[y_series == 0].tolist()
        
        if len(pos_indices) == 0:
            return None
        
        # Get unique categorical values from positive examples
        pos_values = X.loc[pos_indices, feature].unique().tolist()
        if len(pos_values) == 0:
            return None
        
        # Use Optimize() to select best categorical value
        opt = Optimize()
        value_selected = {}
        
        # Create binary variables for each categorical value
        for val in pos_values:
            # Use a simplified variable name (avoid special characters)
            val_str = str(val).replace("'", "").replace('"', '').replace(' ', '_').replace(',', '_')
            val_var = Int(f'select_{feature}_{val_str}')
            opt.add(val_var >= 0)
            opt.add(val_var <= 1)
            value_selected[val] = val_var
        
        # Constraint: select exactly one value (for single value optimization)
        # This is different from set membership - we want ONE best value
        selected_sum = Sum([value_selected[val] for val in pos_values])
        opt.add(selected_sum == 1)
        
        # Count positives and negatives that match selected value
        pos_satisfied = []
        neg_satisfied = []
        
        for idx in pos_indices:
            feat_val = X.loc[idx, feature]
            if feat_val in value_selected:
                pos_satisfied.append(If(value_selected[feat_val] == 1, 1, 0))
        
        for idx in neg_indices:
            feat_val = X.loc[idx, feature]
            if feat_val in value_selected:
                neg_satisfied.append(If(value_selected[feat_val] == 1, 1, 0))
        
        # Optimize: maximize positive coverage, minimize negative coverage
        if pos_satisfied:
            pos_count = Sum(pos_satisfied) if len(
                pos_satisfied) > 1 else pos_satisfied[0]
            opt.maximize(pos_count)
        
        if neg_satisfied:
            neg_count = Sum(neg_satisfied) if len(
                neg_satisfied) > 1 else neg_satisfied[0]
            opt.minimize(neg_count)
        
        if opt.check() == sat:
            model = opt.model()
            for val, val_var in value_selected.items():
                try:
                    if model[val_var].as_long() == 1:
                        # Decode if needed
                        if feature in self.feature_encoders:
                            try:
                                if pd.api.types.is_numeric_dtype(type(val)) or isinstance(val, (int, float)):
                                    value_int = int(float(val))
                                    encoder = self.feature_encoders[feature]
                                    if hasattr(encoder, 'classes_') and value_int < len(encoder.classes_):
                                        return encoder.classes_[value_int]
                                    else:
                                        return str(val)
                                else:
                                    return str(val)
                            except:
                                return str(val)
                        else:
                            return str(val)
                except:
                    continue
            return None
        else:
            return None
    
    def _generate_categorical_rules_with_z3(self, X: pd.DataFrame, y: np.ndarray) -> List[Dict]:
        """
        INNOVATIVE APPROACH: Generate categorical rules directly with Z3 (like numsynth's magic_value_type)
        
        Instead of letting PyGol learn categorical variables (color(X, B)) and trying to replace them,
        we generate rules with specific categorical values directly and use Z3 to optimize them.
        
        This generates rules like:
        - target(X) :- color(X, 'red')
        - target(X) :- color(X, 'red'), position_x(X, V), V > threshold (optimized by Z3)
        - target(X) :- orientation(X, 'upright'), size(X, V), V < threshold (optimized by Z3)
        
        This is similar to how numsynth uses magic_value_type to tell the system to use
        specific categorical values directly in the hypothesis space.
        """
        if not hasattr(self, 'categorical_features') or not self.categorical_features:
            return []
        
        categorical_rules = []
        y_series = pd.Series(y, index=X.index)
        pos_indices = X.index[y_series == 1].tolist()
        neg_indices = X.index[y_series == 0].tolist()
        
        if len(pos_indices) == 0 or len(neg_indices) == 0:
            return []
        
        # Get numeric features (to combine with categorical)
        numeric_features = [col for col in X.columns 
                           if col not in self.categorical_features 
                           and pd.api.types.is_numeric_dtype(X[col])]
        
        # For each categorical feature and value, generate rules
        for cat_feature in self.categorical_features:
            if cat_feature not in X.columns:
                continue
            
            # Get unique categorical values from POSITIVE examples (more discriminative)
            pos_values = X.loc[pos_indices, cat_feature].unique().tolist()
            
            # Decode values if needed
            decoded_values = []
            for val in pos_values:
                if cat_feature in self.feature_encoders:
                    try:
                        if pd.api.types.is_numeric_dtype(type(val)) or isinstance(val, (int, float)):
                            value_int = int(float(val))
                            encoder = self.feature_encoders[cat_feature]
                            if hasattr(encoder, 'classes_') and value_int < len(encoder.classes_):
                                decoded_values.append(encoder.classes_[value_int])
                            else:
                                decoded_values.append(str(val))
                        else:
                            decoded_values.append(str(val))
                    except:
                        decoded_values.append(str(val))
                else:
                    decoded_values.append(str(val))
            
            # For each categorical value, use Z3 to find if it's discriminative
            for cat_value in decoded_values:
                # Count positives and negatives with this value
                pos_mask = pd.Series(False, index=X.index)
                neg_mask = pd.Series(False, index=X.index)
                
                for idx in pos_indices:
                    feat_val = X.loc[idx, cat_feature]
                    # Decode if needed
                    decoded_feat_val = self._decode_categorical_value(cat_feature, feat_val)
                    if decoded_feat_val == cat_value:
                        pos_mask.loc[idx] = True
                
                for idx in neg_indices:
                    feat_val = X.loc[idx, cat_feature]
                    decoded_feat_val = self._decode_categorical_value(cat_feature, feat_val)
                    if decoded_feat_val == cat_value:
                        neg_mask.loc[idx] = True
                
                pos_count = pos_mask.sum()
                neg_count = neg_mask.sum()
                total_pos = len(pos_indices)
                total_neg = len(neg_indices)
                
                # Calculate precision
                precision = pos_count / max(pos_count + neg_count, 1)
                recall = pos_count / max(total_pos, 1)
                
                # Only keep if precision > 0.5 (better than random)
                if precision > 0.5 and recall > 0.1:
                    # Rule 1: Pure categorical rule
                    rule = {
                        'type': 'single_feature',
                        'feature': cat_feature,
                        'operation': '==',
                        'value': cat_value,
                        'precision': precision,
                        'recall': recall,
                        'f1': 2 * precision * recall / max(precision + recall, 1e-10),
                        'pos_coverage': recall,
                        'neg_exclusion': 1 - (neg_count / max(total_neg, 1)),
                        'support': (pos_count + neg_count) / len(X),
                        'score': 0.4 * precision + 0.3 * recall + 0.2 * (2 * precision * recall / max(precision + recall, 1e-10)),
                        'verified': True,
                        'smt_optimized': False,  # Generated directly, not optimized
                        'categorical_generated': True  # Mark as generated
                    }
                    categorical_rules.append(rule)
                    
                    # Rule 2: Categorical + numeric (use Z3 to optimize threshold)
                    for num_feature in numeric_features[:5]:  # Limit to first 5 to avoid explosion
                        try:
                            # Get values for examples that match the categorical value
                            matching_indices = X.index[pos_mask | neg_mask].tolist()
                            if len(matching_indices) < 10:  # Need enough examples
                                continue
                            
                            # Use Z3 to find optimal threshold for numeric feature
                            # among examples that match the categorical value
                            matching_X = X.loc[matching_indices]
                            matching_y = y_series.loc[matching_indices].values
                            
                            # Try both > and < operations
                            for operation in ['>', '<']:
                                # Get initial threshold (median of matching examples)
                                if operation == '>':
                                    initial_threshold = float(matching_X[num_feature].quantile(0.5))
                                else:
                                    initial_threshold = float(matching_X[num_feature].quantile(0.5))
                                
                                # Optimize threshold using Z3
                                optimized_threshold = self._optimize_threshold_smt(
                                    {'type': 'single_feature', 'feature': num_feature},
                                    matching_X, matching_y, num_feature, operation, initial_threshold
                                )
                                
                                if optimized_threshold is not None:
                                    # Evaluate the combined rule
                                    cat_mask = pos_mask | neg_mask
                                    num_mask = self._apply_numeric_condition(matching_X, num_feature, operation, optimized_threshold)
                                    combined_mask = cat_mask & pd.Series(num_mask, index=matching_X.index)
                                    
                                    pos_covered = combined_mask[pos_mask].sum()
                                    neg_covered = combined_mask[neg_mask].sum()
                                    
                                    combined_precision = pos_covered / max(pos_covered + neg_covered, 1)
                                    combined_recall = pos_covered / max(total_pos, 1)
                                    combined_f1 = 2 * combined_precision * combined_recall / max(combined_precision + combined_recall, 1e-10)
                                    
                                    # Only keep if better than pure categorical rule
                                    if combined_precision > precision and combined_f1 > rule['f1']:
                                        combined_rule = {
                                            'type': 'conjunction',
                                            'components': [
                                                {
                                                    'feature': cat_feature,
                                                    'operation': '==',
                                                    'value': cat_value
                                                },
                                                {
                                                    'feature': num_feature,
                                                    'operation': operation,
                                                    'threshold': optimized_threshold,
                                                    'threshold_optimized': True
                                                }
                                            ],
                                            'precision': combined_precision,
                                            'recall': combined_recall,
                                            'f1': combined_f1,
                                            'pos_coverage': combined_recall,
                                            'neg_exclusion': 1 - (neg_covered / max(total_neg, 1)),
                                            'support': combined_mask.sum() / len(X),
                                            'score': 0.4 * combined_precision + 0.3 * combined_recall + 0.2 * combined_f1,
                                            'verified': True,
                                            'smt_optimized': True,
                                            'categorical_generated': True
                                        }
                                        categorical_rules.append(combined_rule)
                        except Exception as e:
                            if self.verbose:
                                print(f"  Error generating categorical+numeric rule for {cat_feature}={cat_value}, {num_feature}: {e}")
                            continue
        
        if self.verbose:
            print(f"  Generated {len(categorical_rules)} categorical rules with Z3")
        
        return categorical_rules
    
    def _decode_categorical_value(self, feature: str, value) -> str:
        """Helper to decode categorical value from numeric encoding"""
        if feature in self.feature_encoders:
            try:
                if pd.api.types.is_numeric_dtype(type(value)) or isinstance(value, (int, float)):
                    value_int = int(float(value))
                    encoder = self.feature_encoders[feature]
                    if hasattr(encoder, 'classes_') and value_int < len(encoder.classes_):
                        return encoder.classes_[value_int]
                    else:
                        return str(value)
                else:
                    return str(value)
            except:
                return str(value)
        else:
            return str(value)
    
    def _apply_numeric_condition(self, X: pd.DataFrame, feature: str, operation: str, threshold: float) -> pd.Series:
        """Helper to apply numeric condition and return boolean mask"""
        try:
            feature_series = pd.to_numeric(X[feature], errors='coerce').astype(float)
            if operation == '>':
                return feature_series > threshold
            elif operation == '<':
                return feature_series < threshold
            else:
                return pd.Series(False, index=X.index)
        except:
            return pd.Series(False, index=X.index)
    
    def _generate_categorical_value_rules(self, X: pd.DataFrame, y: np.ndarray) -> List[Dict]:
        """
        Generate rules with specific categorical values directly (numsynth-style)
        
        Instead of letting PyGol learn variables and post-processing, generate candidate rules
        with specific categorical values directly, similar to how numsynth uses magic_value_type.
        
        This generates rules like:
        - target(X) :- color(X, 'red')
        - target(X) :- color(X, 'red'), position_x(X, V), V > threshold
        - target(X) :- orientation(X, 'upright'), size(X, V), V < threshold
        
        This mimics numsynth's approach where magic_value_type tells the system to use
        specific categorical values directly in the hypothesis space.
        """
        if not hasattr(self, 'categorical_features') or not self.categorical_features:
            return []
        
        categorical_rules = []
        y_series = pd.Series(y, index=X.index)
        pos_indices = X.index[y_series == 1].tolist()
        neg_indices = X.index[y_series == 0].tolist()
        
        if len(pos_indices) == 0 or len(neg_indices) == 0:
            return []
        
        # For each categorical feature, generate rules with each possible value
        for cat_feature in self.categorical_features:
            if cat_feature not in X.columns:
                continue
            
            # Get unique categorical values from the data
            unique_values = X[cat_feature].unique().tolist()
            
            # Decode values if needed
            decoded_values = []
            for val in unique_values:
                if cat_feature in self.feature_encoders:
                    try:
                        if pd.api.types.is_numeric_dtype(type(val)) or isinstance(val, (int, float)):
                            value_int = int(float(val))
                            encoder = self.feature_encoders[cat_feature]
                            if hasattr(encoder, 'classes_') and value_int < len(encoder.classes_):
                                decoded_values.append(encoder.classes_[value_int])
                            else:
                                decoded_values.append(str(val))
                        else:
                            decoded_values.append(str(val))
                    except:
                        decoded_values.append(str(val))
                else:
                    decoded_values.append(str(val))
            
            # For each categorical value, check if it's discriminative
            for cat_value in decoded_values:
                # Count positives and negatives with this value
                pos_with_value = 0
                neg_with_value = 0
                
                for idx in pos_indices:
                    feat_val = X.loc[idx, cat_feature]
                    # Decode if needed
                    if cat_feature in self.feature_encoders:
                        try:
                            if pd.api.types.is_numeric_dtype(type(feat_val)) or isinstance(feat_val, (int, float)):
                                value_int = int(float(feat_val))
                                encoder = self.feature_encoders[cat_feature]
                                if hasattr(encoder, 'classes_') and value_int < len(encoder.classes_):
                                    decoded_feat_val = encoder.classes_[value_int]
                                else:
                                    decoded_feat_val = str(feat_val)
                            else:
                                decoded_feat_val = str(feat_val)
                        except:
                            decoded_feat_val = str(feat_val)
                    else:
                        decoded_feat_val = str(feat_val)
                    
                    if decoded_feat_val == cat_value:
                        pos_with_value += 1
                
                for idx in neg_indices:
                    feat_val = X.loc[idx, cat_feature]
                    # Decode if needed
                    if cat_feature in self.feature_encoders:
                        try:
                            if pd.api.types.is_numeric_dtype(type(feat_val)) or isinstance(feat_val, (int, float)):
                                value_int = int(float(feat_val))
                                encoder = self.feature_encoders[cat_feature]
                                if hasattr(encoder, 'classes_') and value_int < len(encoder.classes_):
                                    decoded_feat_val = encoder.classes_[value_int]
                                else:
                                    decoded_feat_val = str(feat_val)
                            else:
                                decoded_feat_val = str(feat_val)
                        except:
                            decoded_feat_val = str(feat_val)
                    else:
                        decoded_feat_val = str(feat_val)
                    
                    if decoded_feat_val == cat_value:
                        neg_with_value += 1
                
                total_with_value = pos_with_value + neg_with_value
                if total_with_value == 0:
                    continue
                
                precision = pos_with_value / total_with_value if total_with_value > 0 else 0
                pos_coverage = pos_with_value / len(pos_indices) if len(pos_indices) > 0 else 0
                
                # Only keep if it's discriminative (precision > 0.5 and covers at least 5% of positives)
                if precision > 0.5 and pos_coverage >= 0.05:
                    # Single categorical value rule
                    rule = {
                        'type': 'single_feature',
                        'feature': cat_feature,
                        'operation': '==',
                        'value': cat_value,
                        'threshold': None,
                        'pos_coverage': pos_coverage,
                        'precision': precision,
                        'support': total_with_value / len(X),
                        'score': 0.4 * pos_coverage + 0.4 * precision,
                        'verified': False,  # Will be verified by SMT
                        'smt_optimized': False,
                        'numsynth_style': True  # Mark as numsynth-style rule
                    }
                    categorical_rules.append(rule)
                    
                    # Also generate conjunction rules: categorical value + numerical feature
                    # This mimics numsynth's hybrid approach
                    for num_feature in X.columns:
                        if num_feature in self.categorical_features or num_feature == cat_feature:
                            continue
                        
                        if not pd.api.types.is_numeric_dtype(X[num_feature]):
                            continue
                        
                        # Use SMT to find optimal threshold for this combination
                        # This creates rules like: color(X, 'red'), position_x(X, V), V > threshold
                        try:
                            num_series = pd.to_numeric(X[num_feature], errors='coerce').astype(float)
                            if num_series.isna().all():
                                continue
                            
                            # Filter to examples with this categorical value
                            cat_mask = pd.Series(False, index=X.index)
                            for idx in X.index:
                                feat_val = X.loc[idx, cat_feature]
                                if cat_feature in self.feature_encoders:
                                    try:
                                        if pd.api.types.is_numeric_dtype(type(feat_val)) or isinstance(feat_val, (int, float)):
                                            value_int = int(float(feat_val))
                                            encoder = self.feature_encoders[cat_feature]
                                            if hasattr(encoder, 'classes_') and value_int < len(encoder.classes_):
                                                decoded_feat_val = encoder.classes_[value_int]
                                            else:
                                                decoded_feat_val = str(feat_val)
                                        else:
                                            decoded_feat_val = str(feat_val)
                                    except:
                                        decoded_feat_val = str(feat_val)
                                else:
                                    decoded_feat_val = str(feat_val)
                                
                                if decoded_feat_val == cat_value:
                                    cat_mask.loc[idx] = True
                            
                            if cat_mask.sum() == 0:
                                continue
                            
                            # Find optimal threshold for numerical feature on examples with this categorical value
                            filtered_pos = X.index[cat_mask & (y_series == 1)].tolist()
                            filtered_neg = X.index[cat_mask & (y_series == 0)].tolist()
                            
                            if len(filtered_pos) == 0:
                                continue
                            
                            # Use SMT to optimize threshold
                            opt = Optimize()
                            threshold = Real('threshold')
                            
                            feat_min = float(num_series.min())
                            feat_max = float(num_series.max())
                            opt.add(threshold >= feat_min)
                            opt.add(threshold <= feat_max)
                            
                            pos_satisfied = []
                            neg_satisfied = []
                            
                            for idx in filtered_pos:
                                val = float(pd.to_numeric(X.loc[idx, num_feature], errors='coerce'))
                                if pd.isna(val):
                                    continue
                                feat_val = Real(f'pos_{idx}_{num_feature}')
                                opt.add(feat_val == val)
                                pos_satisfied.append(If(feat_val > threshold, 1, 0))
                            
                            for idx in filtered_neg:
                                val = float(pd.to_numeric(X.loc[idx, num_feature], errors='coerce'))
                                if pd.isna(val):
                                    continue
                                feat_val = Real(f'neg_{idx}_{num_feature}')
                                opt.add(feat_val == val)
                                neg_satisfied.append(If(feat_val > threshold, 1, 0))
                            
                            if pos_satisfied:
                                pos_count = Sum(pos_satisfied) if len(pos_satisfied) > 1 else pos_satisfied[0]
                                opt.maximize(pos_count)
                                
                                if neg_satisfied:
                                    neg_count = Sum(neg_satisfied) if len(neg_satisfied) > 1 else neg_satisfied[0]
                                    opt.minimize(neg_count)
                                
                                if opt.check() == sat:
                                    model = opt.model()
                                    threshold_val = float(model[threshold].as_fraction())
                                    
                                    # Evaluate the conjunction rule
                                    cat_mask_filtered = cat_mask
                                    num_mask = num_series > threshold_val
                                    conj_mask = cat_mask_filtered & num_mask
                                    
                                    pos_covered = conj_mask[y_series == 1].sum()
                                    neg_covered = conj_mask[y_series == 0].sum()
                                    total_pos = (y_series == 1).sum()
                                    total_neg = (y_series == 0).sum()
                                    
                                    conj_pos_coverage = pos_covered / max(total_pos, 1)
                                    conj_precision = pos_covered / max(conj_mask.sum(), 1) if conj_mask.sum() > 0 else 0
                                    
                                    if conj_precision > 0.5 and conj_pos_coverage >= 0.05:
                                        conj_rule = {
                                            'type': 'conjunction',
                                            'components': [
                                                {
                                                    'feature': cat_feature,
                                                    'operation': '==',
                                                    'value': cat_value,
                                                    'threshold': None
                                                },
                                                {
                                                    'feature': num_feature,
                                                    'operation': '>',
                                                    'threshold': threshold_val,
                                                    'threshold_var': None
                                                }
                                            ],
                                            'pos_coverage': conj_pos_coverage,
                                            'precision': conj_precision,
                                            'support': conj_mask.sum() / len(X),
                                            'score': 0.4 * conj_pos_coverage + 0.4 * conj_precision,
                                            'verified': False,
                                            'smt_optimized': True,
                                            'numsynth_style': True
                                        }
                                        categorical_rules.append(conj_rule)
                        except Exception as e:
                            if self.verbose:
                                print(f"  Error generating conjunction rule for {cat_feature}={cat_value}, {num_feature}: {e}")
                            continue
        
        if self.verbose:
            print(f"[Categorical Rule Generation] Generated {len(categorical_rules)} numsynth-style categorical rules")
        
        return categorical_rules
    
    def _optimize_categorical_smt(self, rule: Dict, X: pd.DataFrame, y: np.ndarray, 
                                   feature: str, initial_values: List[str]) -> Optional[List[str]]:
        """
        Categorical threshold learning
        Uses SMT to find optimal set of categorical values that maximize positive coverage
        """
        if feature not in X.columns:
            return initial_values
        
        y_series = pd.Series(y, index=X.index)
        pos_indices = X.index[y_series == 1].tolist()
        neg_indices = X.index[y_series == 0].tolist()
        
        if len(pos_indices) == 0 or len(neg_indices) == 0:
            return initial_values
        
        # Get unique categorical values
        unique_values = X[feature].unique().tolist()
        if len(unique_values) == 0:
            return initial_values
        
        # Use Optimize() to select best categorical values
        opt = Optimize()
        value_selected = {}
        
        # Create binary variables for each categorical value
        for val in unique_values:
            val_var = Int(f'select_{feature}_{val}')
            opt.add(val_var >= 0)
            opt.add(val_var <= 1)
            value_selected[val] = val_var
        
        # Count positives and negatives that match selected values
        pos_satisfied = []
        neg_satisfied = []
        
        for idx in pos_indices:
            feat_val = X.loc[idx, feature]
            if feat_val in value_selected:
                pos_satisfied.append(If(value_selected[feat_val] == 1, 1, 0))
        
        for idx in neg_indices:
            feat_val = X.loc[idx, feature]
            if feat_val in value_selected:
                neg_satisfied.append(If(value_selected[feat_val] == 1, 1, 0))
        
        # Optimize: maximize positive coverage, minimize negative coverage
        if pos_satisfied:
            pos_count = Sum(pos_satisfied) if len(
                pos_satisfied) > 1 else pos_satisfied[0]
            opt.maximize(pos_count)
        
        if neg_satisfied:
            neg_count = Sum(neg_satisfied) if len(
                neg_satisfied) > 1 else neg_satisfied[0]
            opt.minimize(neg_count)
        
        result = opt.check()
        if result == sat:
            model = opt.model()
            selected_values = []
            for val in unique_values:
                try:
                    if model[value_selected[val]].as_long() == 1:
                        selected_values.append(str(val))
                except:
                    pass
            return selected_values if selected_values else initial_values
        else:
            return initial_values
    
    def _compute_arithmetic_feature(
    self,
    feature_name: str,
     X: pd.DataFrame) -> pd.Series:
        """Compute arithmetic features on the fly (sum_x_y, scaled_2_x, etc.)"""
        if self.verbose:
            print(f"  Computing arithmetic feature: {feature_name}")
            print(f"  Available columns: {list(X.columns)}")
        try:
            # Pattern: sum_col1_col2
            if feature_name.startswith('sum_'):
                suffix = feature_name.replace('sum_', '')
                # Try to find matching columns by checking all possible splits
                for col1 in X.columns:
                    if suffix.startswith(col1 + '_'):
                        col2 = suffix.replace(col1 + '_', '', 1)
                        if col2 in X.columns:
                            if self.verbose:
                                print(f"  Found sum: {col1} + {col2}")
                            val1 = pd.to_numeric(
    X[col1], errors='coerce').astype(float).fillna(0.0)
                            val2 = pd.to_numeric(
    X[col2], errors='coerce').astype(float).fillna(0.0)
                            result = val1 + val2
                            if self.verbose:
                                print(f"  Computed sum: {result.head(3).tolist()}")
                            return result

            # Pattern: prod_col1_col2
            elif feature_name.startswith('prod_'):
                suffix = feature_name.replace('prod_', '')
                for col1 in X.columns:
                    if suffix.startswith(col1 + '_'):
                        col2 = suffix.replace(col1 + '_', '', 1)
                        if col2 in X.columns:
                            val1 = pd.to_numeric(
    X[col1], errors='coerce').astype(float).fillna(0.0)
                            val2 = pd.to_numeric(
    X[col2], errors='coerce').astype(float).fillna(0.0)
                            return val1 * val2

            # Pattern: diff_col1_col2
            elif feature_name.startswith('diff_'):
                suffix = feature_name.replace('diff_', '')
                for col1 in X.columns:
                    if suffix.startswith(col1 + '_'):
                        col2 = suffix.replace(col1 + '_', '', 1)
                        if col2 in X.columns:
                            val1 = pd.to_numeric(
    X[col1], errors='coerce').astype(float).fillna(0.0)
                            val2 = pd.to_numeric(
    X[col2], errors='coerce').astype(float).fillna(0.0)
                            return val1 - val2

            # Pattern: scaled_coeff_col (e.g., scaled_2_x, scaled_0.5_y)
            elif feature_name.startswith('scaled_'):
                suffix = feature_name.replace('scaled_', '')
                # Try to extract coefficient and column name
                # Pattern: scaled_2_x or scaled_0.5_x
                parts = suffix.split('_', 1)
                if len(parts) == 2:
                    try:
                        coeff = float(parts[0])
                        col = parts[1]
                        if self.verbose:
                            print(f"  Trying scaled: {coeff} * {col}")
                        if col in X.columns:
                            val = pd.to_numeric(
    X[col], errors='coerce').astype(float).fillna(0.0)
                            result = val * coeff
                            if self.verbose:
                                print(f"  Computed scaled: {result.head(3).tolist()}")
                            return result
                    except ValueError:
                        pass
                # Also try without underscore (e.g., scaled_2x)
                # Extract number from start
                import re
                match = re.match(r'^([0-9.]+)(.+)$', suffix)
                if match:
                    try:
                        coeff = float(match.group(1))
                        col = match.group(2)
                        if self.verbose:
                            print(f"  Trying scaled (no underscore): {coeff} * {col}")
                        if col in X.columns:
                            val = pd.to_numeric(X[col], errors='coerce').astype(float).fillna(0.0)
                            result = val * coeff
                            if self.verbose:
                                print(f"  Computed scaled: {result.head(3).tolist()}")
                            return result
                    except (ValueError, IndexError):
                        pass

            # Pattern: add_fact or mult_fact - these are more complex, skip for now
            # They require multiple arguments which is harder to handle

        except Exception as e:
            if self.verbose:
                print(
    f"  Error computing arithmetic feature {feature_name}: {e}")
                import traceback
                traceback.print_exc()

        if self.verbose:
            print(
    f"  Could not compute arithmetic feature: {feature_name}")
        return None
    
    def _apply_rule(self, rule: Dict, X: pd.DataFrame) -> pd.Series:
        """Apply a rule to get boolean mask"""
        try:
            if rule['type'] == 'arithmetic_linear':
                # Handle arithmetic linear relationships learned by SMT
                # Format: coeff1*feature1 + coeff2*feature2 <= threshold (or 3-variable)
                features = rule['features']
                coefficients = rule['coefficients']
                threshold = rule['threshold']
                operation = rule.get('operation', '<=')

                # Handle 3-variable case
                if len(features) == 3 and len(coefficients) == 3:
                    col1, col2, col3 = features
                    coeff1, coeff2, coeff3 = coefficients
                    
                    if col1 not in X.columns or col2 not in X.columns or col3 not in X.columns:
                        return pd.Series(False, index=X.index)
                    
                    try:
                        val1_series = pd.to_numeric(X[col1], errors='coerce').astype(float).fillna(0.0)
                        val2_series = pd.to_numeric(X[col2], errors='coerce').astype(float).fillna(0.0)
                        val3_series = pd.to_numeric(X[col3], errors='coerce').astype(float).fillna(0.0)
                        
                        linear_comb = coeff1 * val1_series + coeff2 * val2_series + coeff3 * val3_series
                        
                        if operation == '<=':
                            result = linear_comb <= threshold
                        elif operation == '>=':
                            result = linear_comb >= threshold
                        else:
                            result = linear_comb <= threshold  # Default
                        
                        return result.fillna(False)
                    except Exception as e:
                        if self.verbose:
                            print(f"Error applying 3-variable arithmetic rule: {e}")
                        return pd.Series(False, index=X.index)

                # Handle 2-variable case
                if len(features) != 2 or len(coefficients) != 2:
                    return pd.Series(False, index=X.index)

                col1, col2 = features
                coeff1, coeff2 = coefficients

                if col1 not in X.columns or col2 not in X.columns:
                    return pd.Series(False, index=X.index)

                try:
                    val1_series = pd.to_numeric(
    X[col1], errors='coerce').astype(float).fillna(0.0)
                    val2_series = pd.to_numeric(
    X[col2], errors='coerce').astype(float).fillna(0.0)

                    linear_comb = coeff1 * val1_series + coeff2 * val2_series

                    if operation == '<=':
                        result = linear_comb <= threshold
                    elif operation == '>=':
                        result = linear_comb >= threshold
                    else:
                        result = linear_comb <= threshold  # Default

                    return result.fillna(False)
                except Exception as e:
                    if self.verbose:
                        print(f"Error applying arithmetic rule: {e}")
                    return pd.Series(False, index=X.index)

            elif rule['type'] == 'single_feature':
                feature = rule['feature']
                operation = rule['operation']
                
                # Check if feature exists or is an arithmetic feature
                if feature not in X.columns:
                    # Try to compute as arithmetic feature
                    computed_feature = self._compute_arithmetic_feature(
                        feature, X)
                    if computed_feature is not None:
                        # Create a temporary DataFrame with the computed
                        # feature
                        X_temp = X.copy()
                        X_temp[feature] = computed_feature
                        # Recursively call with the temporary DataFrame
                        return self._apply_rule(rule, X_temp)
                    else:
                        return pd.Series(False, index=X.index)

                # Always try to convert to numeric for > and < operations
                # This handles mixed-type columns safely
                if operation in ['>', '<']:
                    try:
                        # Get the feature column
                        feature_col = X[feature]

                        # Convert to numeric and force float dtype
                        # This prevents object dtype issues when column has
                        # mixed types
                        feature_series = pd.to_numeric(
    feature_col, errors='coerce').astype(float)

                        # Check if we got any valid numeric values
                        if feature_series.isna().all():
                            # All values were non-numeric, return False for all
                            return pd.Series(False, index=X.index)

                        # Ensure threshold is float
                        # This prevents Float64DType vs StrDType comparison
                        # errors
                        threshold = rule.get('threshold', 0)
                        try:
                            # Force conversion to float, handling all edge
                            # cases
                            if threshold is None:
                                threshold = 0.0
                            elif isinstance(threshold, str):
                                # Explicitly handle string thresholds
                                threshold = float(threshold)
                            else:
                                threshold = float(threshold)

                            # VERIFY it's actually a float (defensive check)
                            if not isinstance(threshold, (int, float)):
                                if self.verbose:
                                    print(f"ERROR: Threshold {threshold} (type: {type(threshold)}) is not numeric after conversion!")
                                return pd.Series(False, index=X.index)
                        except (TypeError, ValueError) as e:
                            if self.verbose:
                                print(f"Warning: Threshold {threshold} (type: {type(threshold)}) cannot be converted to float: {e}")
                            return pd.Series(False, index=X.index)

                        # Compare float values safely
                        # Double-check types before comparison (defensive
                        # programming)
                        if not isinstance(threshold, (int, float)):
                            if self.verbose:
                                print(
    f"ERROR: Threshold {threshold} is still not numeric before comparison!")
                            return pd.Series(False, index=X.index)

                        # Fill NaN values with False (they don't match the
                        # condition)
                        if operation == '>':
                            result = feature_series > threshold
                            return result.fillna(False)
                        elif operation == '<':
                            result = feature_series < threshold
                            return result.fillna(False)
                    except (TypeError, ValueError) as e:
                        # If conversion fails completely, return False
                        if self.verbose:
                            print(f"Warning: Could not convert feature {feature} to numeric: {e}")
                        return pd.Series(False, index=X.index)
                    except Exception as e:
                        # Catch any other errors (including numpy ufunc errors)
                        if self.verbose:
                            print(f"Warning: Error applying rule to feature {feature}: {e}")
                            import traceback
                            traceback.print_exc()
                        return pd.Series(False, index=X.index)
                elif operation == '==':
                    value = rule['value']
                    # Decode categorical values before comparison
                    if feature in (self.categorical_features or []):
                        # This is a categorical feature - value is a string, X[feature] is encoded
                        if feature in self.feature_encoders:
                            encoder = self.feature_encoders[feature]
                            try:
                                if isinstance(value, str):
                                    # Value is a string like 'red' - encode it
                                    encoded_value = encoder.transform([value])[0]
                                    return X[feature] == encoded_value
                                else:
                                    # Value is already encoded - use directly
                                    return X[feature] == value
                            except (ValueError, KeyError, IndexError):
                                # If encoding fails (value not in encoder), return False
                                return pd.Series(False, index=X.index)
                        else:
                            # No encoder - compare directly (might fail if types don't match)
                            return X[feature] == value
                    else:
                        # Numeric feature - compare directly
                        return X[feature] == value
                elif operation == 'in':
                    # Set membership for categorical features
                    if isinstance(rule.get('value'), list):
                        return X[feature].isin(rule['value'])
                    else:
                        return X[feature] == rule['value']
            elif rule['type'] == 'conjunction':
                mask = pd.Series(True, index=X.index)
                # Create a temporary DataFrame to store computed arithmetic features
                X_temp = X.copy()
                
                for comp in rule['components']:
                    feature = comp['feature']
                    if feature not in X_temp.columns:
                        # Try to compute as arithmetic feature
                        computed_feature = self._compute_arithmetic_feature(feature, X_temp)
                        if computed_feature is not None:
                            X_temp[feature] = computed_feature
                        else:
                            mask &= pd.Series(False, index=X.index)
                            continue
                    
                    op = comp['operation']
                    
                    if op in ['>', '<']:
                        try:
                            # Get the feature column
                            feature_col = X_temp[feature]
                            
                            # Convert to numeric and force float dtype
                            # This prevents object dtype issues when column has mixed types
                            feature_series = pd.to_numeric(feature_col, errors='coerce').astype(float)
                            
                            # Check if we got any valid numeric values
                            if feature_series.isna().all():
                                # All values were non-numeric, set mask to False
                                mask &= pd.Series(False, index=X.index)
                            else:
                                # Ensure threshold is float
                                # This prevents Float64DType vs StrDType comparison errors
                                threshold = comp.get('threshold', 0)
                                try:
                                    # Force conversion to float, handling all edge cases
                                    if threshold is None:
                                        threshold = 0.0
                                    elif isinstance(threshold, str):
                                        # Explicitly handle string thresholds
                                        threshold = float(threshold)
                                    else:
                                        threshold = float(threshold)
                                    
                                    # VERIFY it's actually a float (defensive check)
                                    if not isinstance(threshold, (int, float)):
                                        if self.verbose:
                                            print(f"ERROR: Threshold {threshold} (type: {type(threshold)}) is not numeric after conversion!")
                                        mask &= pd.Series(False, index=X.index)
                                        continue
                                except (TypeError, ValueError) as e:
                                    if self.verbose:
                                        print(f"Warning: Threshold {threshold} (type: {type(threshold)}) cannot be converted to float: {e}")
                                    mask &= pd.Series(False, index=X.index)
                                    continue
                                
                                # Compare float values safely
                                # Double-check types before comparison (defensive programming)
                                if not isinstance(threshold, (int, float)):
                                    if self.verbose:
                                        print(f"ERROR: Threshold {threshold} is still not numeric before comparison!")
                                    mask &= pd.Series(False, index=X.index)
                                    continue
                                
                                # Fill NaN values with False for safe comparison
                                if op == '>':
                                    result = feature_series > threshold
                                    mask &= result.fillna(False)
                                elif op == '<':
                                    result = feature_series < threshold
                                    mask &= result.fillna(False)
                        except (TypeError, ValueError) as e:
                            # If conversion fails, set mask to False
                            if self.verbose:
                                print(f"Warning: Could not convert feature {feature} to numeric: {e}")
                            mask &= pd.Series(False, index=X.index)
                        except Exception as e:
                            # Catch any other errors (including numpy ufunc errors)
                            if self.verbose:
                                print(f"Warning: Error applying conjunction component {feature}: {e}")
                                import traceback
                                traceback.print_exc()
                            mask &= pd.Series(False, index=X.index)
                    elif op == '==':
                        value = comp.get('value', comp.get('threshold', 0))
                        # Decode categorical values before comparison
                        if feature in (self.categorical_features or []):
                            # This is a categorical feature - value is a string, X[feature] is encoded
                            if feature in self.feature_encoders:
                                encoder = self.feature_encoders[feature]
                                try:
                                    if isinstance(value, str):
                                        # Value is a string like 'red' - encode it
                                        encoded_value = encoder.transform([value])[0]
                                        mask &= (X_temp[feature] == encoded_value)
                                    else:
                                        # Value is already encoded - use directly
                                        mask &= (X_temp[feature] == value)
                                except (ValueError, KeyError, IndexError):
                                    # If encoding fails (value not in encoder), set mask to False
                                    mask &= pd.Series(False, index=X.index)
                            else:
                                # No encoder - compare directly (might fail if types don't match)
                                mask &= (X_temp[feature] == value)
                        else:
                            # Numeric feature - compare directly
                            mask &= (X_temp[feature] == value)
                    elif op == 'in':
                        # Set membership for categorical features
                        if isinstance(comp.get('value'), list):
                            mask &= X_temp[feature].isin(comp['value'])
                        else:
                            mask &= X_temp[feature] == comp['value']
                    elif op == 'range':
                        # Range component: lower < feature < upper
                        lower = comp.get('lower_bound', 0)
                        upper = comp.get('upper_bound', 0)
                        try:
                            feature_series = pd.to_numeric(X_temp[feature], errors='coerce').astype(float)
                            result = (feature_series > lower) & (feature_series < upper)
                            mask &= result.fillna(False)
                        except Exception as e:
                            if self.verbose:
                                print(f"Warning: Error applying range component for {feature}: {e}")
                            mask &= pd.Series(False, index=X.index)
                    elif op == 'arithmetic_linear':
                        # Arithmetic component: coeff1*x + coeff2*y <= threshold
                        features = comp.get('features', [])
                        coefficients = comp.get('coefficients', [])
                        threshold = comp.get('threshold', 0)
                        if len(features) == 2 and len(coefficients) == 2:
                            try:
                                val1_series = pd.to_numeric(X_temp[features[0]], errors='coerce').astype(float)
                                val2_series = pd.to_numeric(X_temp[features[1]], errors='coerce').astype(float)
                                linear_comb = coefficients[0] * val1_series + coefficients[1] * val2_series
                                result = linear_comb <= threshold
                                mask &= result.fillna(False)
                            except Exception as e:
                                if self.verbose:
                                    print(f"Warning: Error applying arithmetic component: {e}")
                                mask &= pd.Series(False, index=X.index)
                return mask
            
            elif rule['type'] == 'range':
                # Range rule for interval problems: lower < feature < upper
                feature = rule.get('feature', '')
                lower = rule.get('lower_bound', 0)
                upper = rule.get('upper_bound', 0)
                
                if feature not in X.columns:
                    return pd.Series(False, index=X.index)
                
                try:
                    feature_series = pd.to_numeric(X[feature], errors='coerce').astype(float)
                    mask = (feature_series > lower) & (feature_series < upper)
                    return mask.fillna(False)
                except Exception as e:
                    if self.verbose:
                        print(f"Error applying range rule: {e}")
                    return pd.Series(False, index=X.index)
        
        except Exception as e:
            # Catch any unexpected errors and return False mask
            if self.verbose:
                print(f"Warning: Unexpected error in _apply_rule: {e}")
                import traceback
                traceback.print_exc()
        return pd.Series(False, index=X.index)
    
    def _add_blocking_constraint(self, rule: Dict):
        """
        Add blocking constraint for UNSAT rule
        Prevents similar bad rules from being generated in next iteration
        """
        try:
            if rule['type'] == 'single_feature':
                feature = rule.get('feature')
                operation = rule.get('operation')
                if operation in ['>', '<']:
                    threshold = rule.get('threshold', 0)
                    # Create blocking constraint: Not(feature > threshold) or Not(feature < threshold)
                    # This will be applied in next iteration's optimization
                    blocking = {
                        'type': 'single_feature',
                        'feature': feature,
                        'operation': operation,
                        'threshold': threshold,
                        'constraint_type': 'blocking'
                    }
                    self.blocking_constraints.append(blocking)
            elif rule['type'] == 'conjunction':
                components = rule.get('components', [])
                if components:
                    blocking = {
                        'type': 'conjunction',
                        'components': components,
                        'constraint_type': 'blocking'
                    }
                    self.blocking_constraints.append(blocking)
        except Exception as e:
            if self.verbose:
                print(f"Error creating blocking constraint: {e}")
    
    def _apply_blocking_constraints(self, opt: Optimize, feature: str, threshold_var, operation: str):
        """
        Apply stored blocking constraints to optimization
        Prevents regenerating similar UNSAT rules
        """
        for blocking in self.blocking_constraints:
            try:
                if blocking.get('type') == 'single_feature' and blocking.get('feature') == feature:
                    blocking_op = blocking.get('operation')
                    blocking_thresh = blocking.get('threshold')
                    if blocking_op == operation:
                        # Block this specific threshold value
                        opt.add(threshold_var != blocking_thresh)
            except Exception:
                continue
    
    def _check_rule_consistency(self, rules: List[Dict], X: pd.DataFrame) -> List[Dict]:
        """
        Phase 2.2: Theory consistency checking
        Checks if rules contradict each other and removes inconsistent rules
        """
        if not rules:
            return []
        
        consistent_rules = []
        
        for i, rule1 in enumerate(rules):
            is_consistent = True
            
            # Check against all other rules
            for j, rule2 in enumerate(rules):
                if i == j:
                    continue
                
                # Check if rules contradict each other
                if self._rules_contradict(rule1, rule2, X):
                    is_consistent = False
                    if self.verbose:
                        print(f"  Rules {i} and {j} contradict, removing rule {i}")
                    break
            
            if is_consistent:
                consistent_rules.append(rule1)
        
        return consistent_rules if consistent_rules else rules  # Fallback to all if empty
    
    def _rules_contradict(self, rule1: Dict, rule2: Dict, X: pd.DataFrame) -> bool:
        """
        Phase 2.2: Check if two rules contradict each other
        Rules contradict if they make opposite predictions on the same examples
        """
        try:
            mask1 = self._apply_rule(rule1, X)
            mask2 = self._apply_rule(rule2, X)
            
            # Check if rules have overlapping coverage but opposite predictions
            # This is a simplified check - full contradiction would require checking
            # if rule1 says "target" and rule2 says "not target" for same examples
            # For now, we check if they cover disjoint sets (which might indicate contradiction)
            
            # More sophisticated: check if rules have same features but opposite thresholds
            if rule1.get('type') == 'single_feature' and rule2.get('type') == 'single_feature':
                if rule1.get('feature') == rule2.get('feature'):
                    op1 = rule1.get('operation')
                    op2 = rule2.get('operation')
                    thresh1 = rule1.get('threshold', rule1.get('value'))
                    thresh2 = rule2.get('threshold', rule2.get('value'))
                    
                    # Contradiction: same feature, opposite operations, overlapping thresholds
                    if (op1 == '>' and op2 == '<') or (op1 == '<' and op2 == '>'):
                        # Check if thresholds overlap (contradiction possible)
                        if isinstance(thresh1, (int, float)) and isinstance(thresh2, (int, float)):
                            if op1 == '>' and op2 == '<':
                                if thresh1 < thresh2:  # x > t1 and x < t2 where t1 < t2
                                    return True  # Contradiction
                            elif op1 == '<' and op2 == '>':
                                if thresh1 > thresh2:  # x < t1 and x > t2 where t1 > t2
                                    return True  # Contradiction
            
            return False
        except Exception:
            return False  # If check fails, assume no contradiction
    
    def _pareto_filter_rules(self, rules: List[Dict]) -> List[Dict]:
        """
        Phase 2.1: Pareto-optimal filtering
        Removes rules that are dominated by others in all objectives
        """
        if not rules:
            return []
        
        pareto_optimal = []
        
        for rule in rules:
            is_dominated = False
            objectives = rule.get('objectives', {})
            
            for other_rule in rules:
                if rule == other_rule:
                    continue
                
                other_objectives = other_rule.get('objectives', {})
                
                # Check if other_rule dominates rule
                # Dominated if other is better or equal in all objectives and strictly better in at least one
                better_or_equal = True
                strictly_better = False
                
                for obj_name in ['pos_coverage', 'neg_exclusion', 'precision', 'recall', 'f1']:
                    if obj_name in objectives and obj_name in other_objectives:
                        if other_objectives[obj_name] < objectives[obj_name]:
                            better_or_equal = False
                            break
                        elif other_objectives[obj_name] > objectives[obj_name]:
                            strictly_better = True
                
                if better_or_equal and strictly_better:
                    is_dominated = True
                    break
            
            if not is_dominated:
                pareto_optimal.append(rule)
        
        return pareto_optimal if pareto_optimal else rules  # Fallback to all if empty
    
    def _calculate_rule_quality(self, rules: List[Dict]) -> float:
        """Calculate overall quality metric for a set of rules
        Uses weighted combination of precision, recall, and accuracy
        Prioritizes precision to reduce false positives and improve accuracy
        """
        if not rules:
            return 0.0
        
        # Calculate metrics from rules
        precisions = [r.get('precision', 0.0) for r in rules if r.get('precision', 0) > 0]
        recalls = [r.get('recall', 0.0) for r in rules if r.get('recall', 0) > 0]
        scores = [r.get('score', 0.0) for r in rules]
        
        # Weighted quality: prioritize precision (reduces false positives)
        # This helps improve accuracy by reducing false positives
        avg_precision = np.mean(precisions) if precisions else 0.0
        avg_recall = np.mean(recalls) if recalls else 0.0
        avg_score = np.mean(scores) if scores else 0.0
        
        # Quality = weighted combination favoring precision for accuracy
        quality = 0.5 * avg_precision + 0.3 * avg_recall + 0.2 * avg_score
        return quality
    
    def fit(self, X: pd.DataFrame, y: np.ndarray):
        """Fit the iterative SMT+ILP learner"""
        # Store training data for use in prediction (e.g., for interval3d bounds)
        self.X_train_ = X.copy()
        self.y_train_ = y.copy()
        try:
            if self.verbose:
                print("Starting iterative SMT+ILP learning")
            
            # Phase 2.3: Cross-validation mode
            if self.use_cross_validation:
                return self._fit_with_cross_validation(X, y)
        
            # Preprocess data - wrap in try/except to catch type errors early
            try:
                X_processed = self._preprocess_data(X)
                # Store preprocessed training data for use in prediction (e.g., for interval3d bounds)
                self.X_train_processed_ = X_processed.copy()
                self.y_train_ = y.copy()
            except Exception as e:
                if self.verbose:
                    print(f"Error in _preprocess_data: {e}")
                    import traceback
                    traceback.print_exc()
                raise
            
            # Phase 2.4: Feature selection via SMT (optional, before learning)
            selected_features = None
            if hasattr(self, 'use_feature_selection') and self.use_feature_selection:
                selected_features = self._select_features_smt(X_processed, y)
                if selected_features:
                    X_processed = X_processed[selected_features]
                if self.verbose:
                    print(f"[Feature Selection] Selected {len(selected_features)} features: {selected_features}")
            
            # Reset background knowledge file (start fresh)
            if os.path.exists(self.bk_file):
                os.remove(self.bk_file)
            
            # Create initial background knowledge
            # Pass original BK file path if available (for geometry problems with arithmetic operations)
            original_bk_file = getattr(self, 'original_bk_file', None)
            self._create_initial_background_knowledge(X_processed, y, original_bk_file=original_bk_file)
        
            # Iterative learning with PyGol and SMT optimization
            all_verified_rules = []
            previous_quality = 0.0
            current_quality = 0.0
            
            # Reset blocking constraints at start
            self.blocking_constraints = []
            
            for iteration in range(self.max_iterations):
                if self.verbose:
                    print(f"\nIteration {iteration + 1}/{self.max_iterations}")
                
                # PyGol learns rules (with variables, as it naturally does)
                try:
                    pygol_rules = self._learn_rules_with_pygol(X_processed, y, iteration + 1)
                    
                    # Learn arithmetic relationships using SMT (like numsynth/Popper)
                    # This happens after PyGol learns simple rules
                    # Use dataset-specific configuration to determine learning strategy
                    dataset_config = getattr(self, 'dataset_config', None)
                    target_pred = getattr(self, 'target_predicate', None)
                    
                    if self.verbose:
                        print(f"[Learning Strategy] Target predicate: {target_pred}")
                        if dataset_config:
                            print(f"[Learning Strategy] Dataset strategy: {dataset_config.get('learning_strategy', 'unknown')}")
                    
                    # Use dataset config to determine learning strategy
                    if dataset_config:
                        learning_strategy = dataset_config.get('learning_strategy', 'simple')
                        use_distance = dataset_config.get('use_distance_learning', False)
                        use_arithmetic = dataset_config.get('use_arithmetic_learning', True)  # Default True for backward compat
                        
                        if learning_strategy == 'arithmetic' and use_arithmetic:
                            # For geometry (halfplane, interval): use arithmetic learning
                            # Geometry needs: halfplane(X,Y) :- Sum is A*X + B*Y, Sum =< Threshold.
                            # Arithmetic learning is primarily for geometry problems
                            arithmetic_rules = self._learn_arithmetic_relationships_smt(pygol_rules, X_processed, y)
                            if arithmetic_rules:
                                # For multihalfplane, ensure we have at least 2 distinct arithmetic rules
                                if target_pred == 'multihalfplane' and len(arithmetic_rules) < 2:
                                    # If we only have 1 rule, try to create a second distinct rule
                                    # by modifying the first rule's coefficients/threshold slightly
                                    if len(arithmetic_rules) == 1:
                                        first_rule = arithmetic_rules[0]
                                        coeffs = first_rule.get('coefficients', [])
                                        threshold = first_rule.get('threshold', 0)
                                        
                                        # Create a second rule with different coefficients
                                        # Try to find a complementary constraint
                                        # Use different coefficient ratios to ensure distinct constraint
                                        if len(coeffs) >= 2:
                                            # Create second rule with swapped/negated coefficients or different ratio
                                            # Option 1: Swap coefficients and adjust threshold
                                            new_coeffs = [coeffs[1], coeffs[0]]  # Swap x and y coefficients
                                            # Adjust threshold based on data distribution
                                            new_threshold = threshold * 1.5  # Increase threshold by 50%
                                            
                                            # Evaluate if this second rule is useful
                                            val1_series = pd.to_numeric(X_processed.iloc[:, 0], errors='coerce').astype(float)
                                            val2_series = pd.to_numeric(X_processed.iloc[:, 1], errors='coerce').astype(float)
                                            linear_comb_2 = new_coeffs[0] * val1_series + new_coeffs[1] * val2_series
                                            mask_2 = linear_comb_2 <= new_threshold
                                            
                                            y_series = pd.Series(y, index=X_processed.index)
                                            pos_covered_2 = mask_2[y_series == 1].sum()
                                            neg_covered_2 = mask_2[y_series == 0].sum()
                                            total_pos = (y_series == 1).sum()
                                            total_neg = (y_series == 0).sum()
                                            
                                            pos_coverage_2 = pos_covered_2 / max(total_pos, 1)
                                            precision_2 = pos_covered_2 / max(mask_2.sum(), 1) if mask_2.sum() > 0 else 0
                                            
                                            if precision_2 > 0.4 and pos_coverage_2 > 0.2:
                                                second_rule = {
                                                    'type': 'arithmetic_linear',
                                                    'features': first_rule.get('features', []),
                                                    'coefficients': new_coeffs,
                                                    'threshold': new_threshold,
                                                    'operation': '<=',
                                                    'pos_coverage': pos_coverage_2,
                                                    'neg_exclusion': 1 - (neg_covered_2 / max(total_neg, 1)),
                                                    'precision': precision_2,
                                                    'recall': pos_coverage_2,
                                                    'f1': 2 * precision_2 * pos_coverage_2 / (precision_2 + pos_coverage_2) if (precision_2 + pos_coverage_2) > 0 else 0,
                                                    'support': mask_2.sum() / len(X_processed),
                                                    'score': 0.4 * pos_coverage_2 + 0.3 * (1 - (neg_covered_2 / max(total_neg, 1))) + 0.2 * precision_2,
                                                    'verified': True,
                                                    'smt_optimized': True
                                                }
                                                arithmetic_rules.append(second_rule)
                                                if self.verbose:
                                                    print(f"Created second arithmetic rule for multihalfplane: "
                                                          f"{new_coeffs[0]:.4f}*{first_rule.get('features', [])[0]} + {new_coeffs[1]:.4f}*{first_rule.get('features', [])[1]} <= {new_threshold:.4f}")
                                
                                pygol_rules.extend(arithmetic_rules)
                                if self.verbose:
                                    print(f"Added {len(arithmetic_rules)} arithmetic relationships to rules")
                                
                                # For multihalfplane: try to create a combined rule with 2 arithmetic constraints
                                # Multiple Halfplanes requires: (a1*x + b1*y <= c1) AND (a2*x + b2*y <= c2)
                                if target_pred == 'multihalfplane' and len(arithmetic_rules) >= 2:
                                    # Find the 2 best arithmetic rules (by precision/score)
                                    # These should be the two halfplane constraints
                                    sorted_arithmetic = sorted(arithmetic_rules, key=lambda r: r.get('score', 0), reverse=True)
                                    rule1 = sorted_arithmetic[0]
                                    rule2 = sorted_arithmetic[1]
                                    
                                    # Create a combined multihalfplane rule
                                    combined_components = [
                                        {
                                            'type': 'arithmetic_linear',
                                            'features': rule1.get('features', []),
                                            'coefficients': rule1.get('coefficients', []),
                                            'threshold': rule1.get('threshold', 0),
                                            'operation': rule1.get('operation', '<=')
                                        },
                                        {
                                            'type': 'arithmetic_linear',
                                            'features': rule2.get('features', []),
                                            'coefficients': rule2.get('coefficients', []),
                                            'threshold': rule2.get('threshold', 0),
                                            'operation': rule2.get('operation', '<=')
                                        }
                                    ]
                                    combined_rule = {
                                        'type': 'conjunction',
                                        'components': combined_components,
                                        'verified': True,
                                        'smt_optimized': True
                                    }
                                    
                                    # Calculate actual precision/recall for combined rule
                                    try:
                                        combined_mask = self._apply_rule(combined_rule, X_processed)
                                        y_series = pd.Series(y, index=X_processed.index)
                                        pos_covered = combined_mask[y_series == 1].sum()
                                        neg_covered = combined_mask[y_series == 0].sum()
                                        total_pos = (y_series == 1).sum()
                                        total_neg = (y_series == 0).sum()
                                        
                                        pos_coverage = pos_covered / max(total_pos, 1)
                                        precision = pos_covered / max(combined_mask.sum(), 1) if combined_mask.sum() > 0 else 0
                                        recall = pos_coverage
                                        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                                        
                                        # Only keep if it's better than individual rules
                                        if precision > 0.5 and recall > 0.1:
                                            combined_rule['precision'] = precision
                                            combined_rule['recall'] = recall
                                            combined_rule['f1'] = f1
                                            combined_rule['pos_coverage'] = pos_coverage
                                            combined_rule['neg_exclusion'] = 1 - (neg_covered / max(total_neg, 1))
                                            combined_rule['score'] = 0.4 * pos_coverage + 0.3 * combined_rule['neg_exclusion'] + 0.3 * precision
                                            pygol_rules.append(combined_rule)
                                            if self.verbose:
                                                print(f"Created combined multihalfplane rule with 2 arithmetic constraints "
                                                      f"(precision={precision:.2f}, recall={recall:.2f}, f1={f1:.2f})")
                                    except Exception as e:
                                        if self.verbose:
                                            print(f"Error evaluating combined multihalfplane rule: {e}")
                        # For 'simple' strategy (list problems), don't add arithmetic or distance rules
                    else:
                        # Fallback: use arithmetic learning for geometry
                        arithmetic_rules = self._learn_arithmetic_relationships_smt(pygol_rules, X_processed, y)
                        if arithmetic_rules:
                            pygol_rules.extend(arithmetic_rules)
                    
                    # Learn range relationships for interval problems (lower < X < upper)
                    # Enable range learning for interval, interval3d, and conjunction problems
                    if target_pred in ['interval', 'interval3d', 'conjunction']:
                        range_rules = self._learn_range_relationships_smt(pygol_rules, X_processed, y)
                        if range_rules:
                            # Add range rules to the list
                            pygol_rules.extend(range_rules)
                            if self.verbose:
                                print(f"Added {len(range_rules)} range relationships to rules")
                            
                            # For interval3d: combine multiple range rules into a single conjunction rule
                            if target_pred == 'interval3d' and len(range_rules) >= 3:
                                # Find x, y, z range rules
                                x_range = next((r for r in range_rules if r.get('feature') == 'x'), None)
                                y_range = next((r for r in range_rules if r.get('feature') == 'y'), None)
                                z_range = next((r for r in range_rules if r.get('feature') == 'z'), None)
                                
                                if x_range and y_range and z_range:
                                    # Combine x, y, z range rules into one conjunction rule
                                    combined_components = [
                                        {
                                            'feature': 'x',
                                            'operation': 'range',
                                            'lower_bound': x_range['lower_bound'],
                                            'upper_bound': x_range['upper_bound']
                                        },
                                        {
                                            'feature': 'y',
                                            'operation': 'range',
                                            'lower_bound': y_range['lower_bound'],
                                            'upper_bound': y_range['upper_bound']
                                        },
                                        {
                                            'feature': 'z',
                                            'operation': 'range',
                                            'lower_bound': z_range['lower_bound'],
                                            'upper_bound': z_range['upper_bound']
                                        }
                                    ]
                                    combined_rule = {
                                        'type': 'conjunction',
                                        'components': combined_components,
                                        'verified': True,
                                        'smt_optimized': True
                                    }
                                    # Calculate precision/recall for combined rule
                                    # The combined rule requires ALL 3 components to match (AND logic)
                                    # So we need to evaluate it on the data to get correct metrics
                                    try:
                                        combined_mask = self._apply_rule(combined_rule, X_processed)
                                        y_series = pd.Series(y, index=X_processed.index)
                                        pos_covered = combined_mask[y_series == 1].sum()
                                        neg_covered = combined_mask[y_series == 0].sum()
                                        total_pos = (y_series == 1).sum()
                                        total_neg = (y_series == 0).sum()
                                        
                                        pos_coverage = pos_covered / max(total_pos, 1)
                                        precision = pos_covered / max(combined_mask.sum(), 1) if combined_mask.sum() > 0 else 0
                                        recall = pos_coverage
                                        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                                        
                                        # Only keep if it's better than individual rules
                                        if precision > 0.5 and recall > 0.1:
                                            combined_rule['precision'] = precision
                                            combined_rule['recall'] = recall
                                            combined_rule['f1'] = f1
                                            combined_rule['pos_coverage'] = pos_coverage
                                            combined_rule['neg_exclusion'] = 1 - (neg_covered / max(total_neg, 1))
                                            combined_rule['score'] = 0.4 * pos_coverage + 0.3 * combined_rule['neg_exclusion'] + 0.3 * precision
                                            pygol_rules.append(combined_rule)
                                            if self.verbose:
                                                print(f"Created combined 3D interval rule with {len(combined_components)} components "
                                                      f"(precision={precision:.2f}, recall={recall:.2f}, f1={f1:.2f})")
                                    except Exception as e:
                                        if self.verbose:
                                            print(f"Error evaluating combined 3D interval rule: {e}")
                                        # Fallback to min() calculation if evaluation fails
                                        combined_rule['precision'] = min(x_range['precision'], y_range['precision'], z_range['precision'])
                                        combined_rule['recall'] = min(x_range['recall'], y_range['recall'], z_range['recall'])
                                        combined_rule['f1'] = min(x_range['f1'], y_range['f1'], z_range['f1'])
                                        combined_rule['score'] = (x_range['score'] + y_range['score'] + z_range['score']) / 3
                                        pygol_rules.append(combined_rule)
                                        if self.verbose:
                                            print(f"Created combined 3D interval rule with {len(combined_components)} components (fallback metrics)")
                            
                            elif target_pred == 'conjunction':
                                # Find arithmetic rule for x+y and range rule for z
                                arithmetic_rule = None
                                z_range_rule = next((r for r in range_rules if r.get('feature') == 'z'), None)
                                
                                # Look for arithmetic rule in existing pygol_rules
                                # For conjunction, we need x+y <= threshold, so look for rules with x and y
                                # Prefer 2-variable rules ['x', 'y'], but also accept 3-variable ['x', 'y', 'z']
                                # If we have 3-variable, we can extract just the x and y coefficients
                                best_rule = None
                                best_score = -1
                                for r in pygol_rules:
                                    if r.get('type') == 'arithmetic_linear':
                                        features = r.get('features', [])
                                        if 'x' in features and 'y' in features:
                                            # Prefer 2-variable rules, but accept 3-variable if no 2-variable found
                                            if len(features) == 2:
                                                arithmetic_rule = r
                                                break  # Found perfect match, use it
                                            elif len(features) == 3 and best_rule is None:
                                                # Store as backup if we don't find a 2-variable rule
                                                # Use the rule with best precision/score
                                                rule_score = r.get('precision', 0) * r.get('recall', 0)
                                                if rule_score > best_score:
                                                    best_rule = r
                                                    best_score = rule_score
                                
                                # If no 2-variable rule found, use the best 3-variable rule
                                if arithmetic_rule is None and best_rule is not None:
                                    # Extract x and y components from 3-variable rule
                                    # For rule: a*x + b*y + c*z <= threshold
                                    # We can use: a*x + b*y <= threshold (ignoring z term)
                                    # But this is approximate - better to learn a specific x+y rule
                                    # For now, try to create a 2-variable rule from the 3-variable one
                                    features_3d = best_rule.get('features', [])
                                    coeffs_3d = best_rule.get('coefficients', [])
                                    threshold_3d = best_rule.get('threshold', 0)
                                    
                                    # Find indices of x and y
                                    x_idx = features_3d.index('x') if 'x' in features_3d else -1
                                    y_idx = features_3d.index('y') if 'y' in features_3d else -1
                                    
                                    if x_idx >= 0 and y_idx >= 0:
                                        # Create a 2-variable rule using x and y coefficients
                                        # Adjust threshold to account for z term (use median z value as approximation)
                                        # Actually, better approach: learn a new x+y rule specifically
                                        # But for now, use the 3-variable rule and let the range rule handle z
                                        arithmetic_rule = {
                                            'type': 'arithmetic_linear',
                                            'features': ['x', 'y'],
                                            'coefficients': [coeffs_3d[x_idx], coeffs_3d[y_idx]],
                                            'threshold': threshold_3d,  # Approximate - z term ignored
                                            'precision': best_rule.get('precision', 0),
                                            'recall': best_rule.get('recall', 0),
                                            'f1': best_rule.get('f1', 0),
                                            'score': best_rule.get('score', 0),
                                            'verified': True,
                                            'smt_optimized': True
                                        }
                                        if self.verbose:
                                            print(f"  [Conjunction] Extracted x+y rule from 3-variable rule: "
                                                  f"{coeffs_3d[x_idx]:.4f}*x + {coeffs_3d[y_idx]:.4f}*y <= {threshold_3d:.4f}")
                                
                                # If still no rule found, try to learn a specific x+y rule
                                if arithmetic_rule is None and z_range_rule:
                                    if self.verbose:
                                        print(f"  [Conjunction] No x+y arithmetic rule found, attempting to learn one...")
                                    # Learn a specific x+y rule for conjunction
                                    try:
                                        y_series = pd.Series(y, index=X_processed.index)
                                        pos_indices = X_processed.index[y_series == 1].tolist()
                                        neg_indices = X_processed.index[y_series == 0].tolist()
                                        
                                        if len(pos_indices) > 0 and len(neg_indices) > 0:
                                            opt = Optimize()
                                            coeff1 = Real('coeff1')
                                            coeff2 = Real('coeff2')
                                            threshold = Real('threshold')
                                            
                                            # Use same bounds as other arithmetic learning
                                            MIN_COEFF, MAX_COEFF = -100.0, 100.0
                                            opt.add(coeff1 >= MIN_COEFF, coeff1 <= MAX_COEFF)
                                            opt.add(coeff2 >= MIN_COEFF, coeff2 <= MAX_COEFF)
                                            opt.add(Or(coeff1 != 0, coeff2 != 0))
                                            
                                            # Get feature ranges
                                            x_series = pd.to_numeric(X_processed['x'], errors='coerce').astype(float)
                                            y_series_feat = pd.to_numeric(X_processed['y'], errors='coerce').astype(float)
                                            
                                            x_min, x_max = float(x_series.min()), float(x_series.max())
                                            y_min, y_max = float(y_series_feat.min()), float(y_series_feat.max())
                                            
                                            max_linear = MAX_COEFF * max(abs(x_min), abs(x_max)) + MAX_COEFF * max(abs(y_min), abs(y_max))
                                            opt.add(threshold >= -max_linear)
                                            opt.add(threshold <= max_linear)
                                            
                                            pos_satisfied = []
                                            neg_soft_constraints = []
                                            
                                            for idx in pos_indices:
                                                v1 = float(pd.to_numeric(X_processed.loc[idx, 'x'], errors='coerce'))
                                                v2 = float(pd.to_numeric(X_processed.loc[idx, 'y'], errors='coerce'))
                                                if pd.isna(v1) or pd.isna(v2):
                                                    continue
                                                
                                                linear_val = Real(f'pos_{idx}_xy')
                                                opt.add(linear_val == coeff1 * v1 + coeff2 * v2)
                                                pos_satisfied.append(If(linear_val <= threshold, 1, 0))
                                            
                                            for idx in neg_indices:
                                                v1 = float(pd.to_numeric(X_processed.loc[idx, 'x'], errors='coerce'))
                                                v2 = float(pd.to_numeric(X_processed.loc[idx, 'y'], errors='coerce'))
                                                if pd.isna(v1) or pd.isna(v2):
                                                    continue
                                                
                                                linear_val = Real(f'neg_{idx}_xy')
                                                opt.add(linear_val == coeff1 * v1 + coeff2 * v2)
                                                neg_soft_constraints.append(linear_val > threshold)
                                            
                                            if pos_satisfied:
                                                opt.maximize(Sum(pos_satisfied))
                                                for neg_constraint in neg_soft_constraints:
                                                    opt.add_soft(neg_constraint, weight=1)
                                                
                                                if opt.check() == sat:
                                                    model = opt.model()
                                                    coeff1_val = float(model[coeff1].as_fraction())
                                                    coeff2_val = float(model[coeff2].as_fraction())
                                                    threshold_val = float(model[threshold].as_fraction())
                                                    
                                                    # Evaluate the rule
                                                    linear_comb = coeff1_val * x_series + coeff2_val * y_series_feat
                                                    mask = linear_comb <= threshold_val
                                                    
                                                    pos_covered = mask[y_series == 1].sum()
                                                    neg_covered = mask[y_series == 0].sum()
                                                    total_pos = (y_series == 1).sum()
                                                    
                                                    pos_coverage = pos_covered / max(total_pos, 1)
                                                    precision = pos_covered / max(mask.sum(), 1) if mask.sum() > 0 else 0
                                                    recall = pos_coverage
                                                    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                                                    
                                                    if precision > 0.5:
                                                        arithmetic_rule = {
                                                            'type': 'arithmetic_linear',
                                                            'features': ['x', 'y'],
                                                            'coefficients': [coeff1_val, coeff2_val],
                                                            'threshold': threshold_val,
                                                            'operation': '<=',
                                                            'pos_coverage': pos_coverage,
                                                            'neg_exclusion': 1 - (neg_covered / max((y_series == 0).sum(), 1)),
                                                            'precision': precision,
                                                            'recall': recall,
                                                            'f1': f1,
                                                            'support': mask.sum() / len(X_processed),
                                                            'score': 0.4 * pos_coverage + 0.3 * (1 - (neg_covered / max((y_series == 0).sum(), 1))) + 0.2 * precision,
                                                            'verified': True,
                                                            'smt_optimized': True
                                                        }
                                                        if self.verbose:
                                                            print(f"  [Conjunction] Learned x+y rule: {coeff1_val:.4f}*x + {coeff2_val:.4f}*y <= {threshold_val:.4f} "
                                                                  f"(precision={precision:.2f}, recall={recall:.2f})")
                                    except Exception as e:
                                        if self.verbose:
                                            print(f"  [Conjunction] Error learning x+y rule: {e}")
                                
                                if arithmetic_rule and z_range_rule:
                                    # Create combined conjunction rule
                                    combined_components = [
                                        {
                                            'feature': 'arithmetic',
                                            'operation': 'arithmetic_linear',
                                            'features': arithmetic_rule['features'],
                                            'coefficients': arithmetic_rule['coefficients'],
                                            'threshold': arithmetic_rule['threshold']
                                        },
                                        {
                                            'feature': 'z',
                                            'operation': 'range',
                                            'lower_bound': z_range_rule['lower_bound'],
                                            'upper_bound': z_range_rule['upper_bound']
                                        }
                                    ]
                                    # Calculate precision/recall for combined rule
                                    # The combined rule requires BOTH components to match (AND logic)
                                    # So we need to evaluate it on the data to get correct metrics
                                    combined_rule = {
                                        'type': 'conjunction',
                                        'components': combined_components,
                                        'verified': True,
                                        'smt_optimized': True
                                    }
                                    # Evaluate the combined rule to get actual metrics
                                    try:
                                        combined_mask = self._apply_rule(combined_rule, X_processed)
                                        y_series = pd.Series(y, index=X_processed.index)
                                        pos_covered = combined_mask[y_series == 1].sum()
                                        neg_covered = combined_mask[y_series == 0].sum()
                                        total_pos = (y_series == 1).sum()
                                        total_neg = (y_series == 0).sum()
                                        
                                        pos_coverage = pos_covered / max(total_pos, 1)
                                        precision = pos_covered / max(combined_mask.sum(), 1) if combined_mask.sum() > 0 else 0
                                        recall = pos_coverage
                                        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                                        
                                        # Only keep if it's better than individual rules
                                        if precision > 0.5 and recall > 0.1:
                                            combined_rule['precision'] = precision
                                            combined_rule['recall'] = recall
                                            combined_rule['f1'] = f1
                                            combined_rule['pos_coverage'] = pos_coverage
                                            combined_rule['neg_exclusion'] = 1 - (neg_covered / max(total_neg, 1))
                                            combined_rule['score'] = 0.4 * pos_coverage + 0.3 * combined_rule['neg_exclusion'] + 0.3 * precision
                                            pygol_rules.append(combined_rule)
                                            if self.verbose:
                                                print(f"Created combined conjunction rule with arithmetic and range components "
                                                      f"(precision={precision:.2f}, recall={recall:.2f}, f1={f1:.2f})")
                                    except Exception as e:
                                        if self.verbose:
                                            print(f"Error evaluating combined conjunction rule: {e}")
                                        # Fallback to min() calculation if evaluation fails
                                        combined_rule['precision'] = min(arithmetic_rule['precision'], z_range_rule['precision'])
                                        combined_rule['recall'] = min(arithmetic_rule['recall'], z_range_rule['recall'])
                                        combined_rule['f1'] = min(arithmetic_rule['f1'], z_range_rule['f1'])
                                        combined_rule['score'] = (arithmetic_rule['score'] + z_range_rule['score']) / 2
                                        pygol_rules.append(combined_rule)
                                        if self.verbose:
                                            print(f"Created combined conjunction rule with arithmetic and range components (fallback metrics)")
                except Exception as e:
                    if self.verbose:
                        print(f"Error in _learn_rules_with_pygol (iteration {iteration + 1}): {e}")
                        import traceback
                        traceback.print_exc()
                    # Continue to next iteration
                    pygol_rules = []
                
                if not pygol_rules:
                    if self.verbose:
                        print("No rules learned, stopping iteration")
                    continue  # Skip to next iteration instead of breaking
                
                # SMT verifies and prunes
                try:
                    verified_rules = self._verify_rules_with_smt(pygol_rules, X_processed, y)
                except Exception as e:
                    if self.verbose:
                        print(f"Error in _verify_rules_with_smt (iteration {iteration + 1}): {e}")
                        import traceback
                        traceback.print_exc()
                    # Continue to next iteration
                    verified_rules = []
                
                if not verified_rules:
                    if self.verbose:
                        print("No verified rules, skipping iteration")
                    continue  # Skip to next iteration
                
                # Add verified rules to collection
                all_verified_rules.extend(verified_rules)
                
                # Calculate quality improvement
                current_quality = self._calculate_rule_quality(verified_rules)
                quality_improvement = current_quality - previous_quality
                
                if self.verbose:
                    print(f"\nQuality: {current_quality:.4f} (improvement: {quality_improvement:.4f})")
                
                # Add high-precision rules to background knowledge in first 3 iterations
                if iteration < self.max_iterations - 1 and iteration < 3:
                    high_precision_rules = [r for r in verified_rules if r.get('precision', 0) > 0.8]
                    if high_precision_rules:
                        self._add_verified_rules_to_bk(high_precision_rules)
                        if self.verbose:
                            print(f"Added {len(high_precision_rules)} high-precision rules (precision > 0.8) to background knowledge (filtered from {len(verified_rules)})")
                    else:
                        if self.verbose:
                            print(f"Skipping adding rules to BK - no high-precision rules (all have precision <= 0.8)")
                if iteration >= 3:
                    if self.verbose:
                        print(f"Skipping adding rules to BK after iteration 3 to avoid over-generalization")
                
                # Check convergence based on quality improvement
                if iteration >= 1:
                    if quality_improvement < -0.01:
                        if self.verbose:
                            print(f"\nConverged: quality significantly decreased ({quality_improvement:.4f})")
                        break
                    elif quality_improvement < self.convergence_threshold and iteration >= 4:
                        if self.verbose:
                            print(f"\nConverged: improvement ({quality_improvement:.4f}) < threshold ({self.convergence_threshold}) after {iteration+1} iterations")
                        break
                    elif quality_improvement < self.convergence_threshold:
                        if self.verbose:
                            print(f"Small improvement ({quality_improvement:.4f}), continuing... (iteration {iteration+1}/{self.max_iterations})")
            
            previous_quality = current_quality
            self.iteration_history.append({
                'iteration': iteration + 1,
                'rules_learned': len(pygol_rules),
                'rules_verified': len(verified_rules),
                'quality': current_quality
            })
            
            # Check theory consistency
            if self.verbose:
                print(f"\n[Consistency Check] Checking consistency of {len(all_verified_rules)} rules...")
            consistent_rules = self._check_rule_consistency(all_verified_rules, X_processed)
            if self.verbose:
                print(f"  Consistent rules: {len(consistent_rules)} (removed {len(all_verified_rules) - len(consistent_rules)} inconsistent)")
            all_verified_rules = consistent_rules
            
            # Finalize: sort by score and select top rules
            all_verified_rules = sorted(all_verified_rules, key=lambda x: x.get('score', 0.0), reverse=True)
            
            # Phase 2.1: Apply Pareto-optimal filtering (optional)
            if hasattr(self, 'use_pareto_filtering') and self.use_pareto_filtering:
                all_verified_rules = self._pareto_filter_rules(all_verified_rules)
            
            # Remove duplicates and degenerate rules
            seen = set()
            unique_rules = []
            for rule in all_verified_rules:
                if self._is_degenerate_rule(rule):
                    if self.verbose:
                        print(f"  Filtering degenerate rule: {rule.get('type', 'unknown')}")
                    continue
                
                sig = self._rule_signature(rule)
                if sig not in seen:
                    seen.add(sig)
                    unique_rules.append(rule)
        
            # Prioritize arithmetic rules, then PyGol structured rules, then others
            arithmetic_rules = [r for r in unique_rules if r.get('type') == 'arithmetic_linear']
            pygol_structure_rules = [r for r in unique_rules if 'pygol_source' in r and 
                                    (r.get('type') == 'conjunction' and len(r.get('components', [])) > 1)]
            other_rules = [r for r in unique_rules if r.get('type') != 'arithmetic_linear' and 
                          not ('pygol_source' in r and r.get('type') == 'conjunction' and len(r.get('components', [])) > 1)]
            arithmetic_rules = sorted(arithmetic_rules, key=lambda x: x.get('score', 0.0), reverse=True)
            pygol_structure_rules = sorted(pygol_structure_rules, key=lambda x: x.get('score', 0.0), reverse=True)
            other_rules = sorted(other_rules, key=lambda x: x.get('score', 0.0), reverse=True)
            all_rules = arithmetic_rules + pygol_structure_rules + other_rules
            if self.verbose and pygol_structure_rules:
                print(f"  [Rule Prioritization] Found {len(pygol_structure_rules)} PyGol rules with structure (multi-feature)")
            
            # Filter rules by precision and recall to balance generalization
            dataset_type = getattr(self, 'dataset_type', 'geometry')
            if dataset_type == 'geometry':
                target_pred = getattr(self, 'target_predicate', None)
                if target_pred in ['halfplane3d', 'conjunction', 'multihalfplane', 'interval3d']:
                    min_precision = 0.65
                    min_recall = 0.05
                    high_precision_threshold = 0.75
                else:
                    min_precision = 0.7
                    min_recall = 0.05
                    high_precision_threshold = 0.8
            else:
                min_precision = 0.6
                min_recall = 0.05
                high_precision_threshold = 0.7
            
            def f1_score(r):
                p = r.get('precision', 0)
                rec = r.get('recall', 0)
                if p + rec == 0:
                    return 0
                return 2 * (p * rec) / (p + rec)
            high_precision_rules = []
            rules_with_cat_filtered = 0
            for r in all_rules:
                has_cat = False
                if r.get('type') == 'conjunction':
                    comps = r.get('components', [])
                    has_cat = any(c.get('feature') in (self.categorical_features or []) for c in comps)
                elif r.get('type') == 'single_feature':
                    has_cat = r.get('feature') in (self.categorical_features or [])
                
                precision = r.get('precision', 0)
                recall = r.get('recall', 0)
                f1 = f1_score(r)
                
                is_pygol_rule = 'pygol_source' in r
                provides_structure = False
                if is_pygol_rule:
                    if r.get('type') == 'conjunction':
                        num_features = len(r.get('components', []))
                        provides_structure = num_features > 1
                    elif r.get('type') == 'single_feature':
                        provides_structure = True
                
                if dataset_type == 'geometry':
                    target_pred = getattr(self, 'target_predicate', None)
                    if target_pred in ['halfplane3d', 'conjunction', 'multihalfplane', 'interval3d']:
                        should_keep = (precision > min_precision and recall >= min_recall) or \
                           precision > high_precision_threshold or \
                           (f1 > 0.45 and precision > 0.55) or \
                           (is_pygol_rule and provides_structure and precision > 0.5 and recall > 0.05)
                    else:
                        should_keep = (precision > min_precision and recall >= min_recall) or \
                           precision > high_precision_threshold or \
                           (f1 > 0.5 and precision > 0.6) or \
                           (is_pygol_rule and provides_structure and precision > 0.55 and recall > 0.05)
                else:
                    should_keep = (precision > min_precision and recall >= min_recall) or \
                       precision > high_precision_threshold or \
                       (f1 > 0.3 and precision > 0.5) or \
                       (is_pygol_rule and provides_structure and precision > 0.5 and recall > 0.05)
                
                if should_keep:
                    high_precision_rules.append(r)
                elif has_cat:
                    rules_with_cat_filtered += 1
                    if self.verbose:
                        print(f"  [Rule Filtering] Filtered out rule with categorical components: precision={precision:.3f}, recall={recall:.3f}, f1={f1:.3f}")
            
            if rules_with_cat_filtered > 0 and self.verbose:
                print(f"  [Rule Filtering] WARNING: Filtered out {rules_with_cat_filtered} rules with categorical components due to low precision/recall")
            
            if not high_precision_rules:
                min_precision = max(0.5, min_precision - 0.1)
                min_recall = max(0.05, min_recall - 0.05)
                high_precision_rules = [r for r in all_rules 
                                      if (r.get('precision', 0) > min_precision and r.get('recall', 0) >= min_recall) 
                                      or r.get('precision', 0) > high_precision_threshold
                                      or (f1_score(r) > 0.3 and r.get('precision', 0) > 0.5)]
            
            balanced_rules = high_precision_rules
            
            if not balanced_rules:
                balanced_rules = [r for r in all_rules if r.get('precision', 0) > 0.5]
                if self.verbose:
                    print(f"  [Warning] No rules met strict criteria, keeping rules with precision > 0.5")
            
            if self.dataset_type == 'geometry':
                target_pred = getattr(self, 'target_predicate', None)
                if target_pred in ['halfplane3d', 'conjunction', 'multihalfplane', 'interval3d']:
                    max_rules = 8
                else:
                    max_rules = 5
            else:
                max_rules = 20
            
            if len(balanced_rules) > max_rules:
                def f1_score(r):
                    p = r.get('precision', 0)
                    rec = r.get('recall', 0)
                    if p + rec == 0:
                        return 0
                    return 2 * (p * rec) / (p + rec)
                
                # Sort by F1 score, but also require minimum F1 > 0.1 to avoid useless rules
                min_f1 = 0.1
                scored_rules = [(r, f1_score(r)) for r in balanced_rules]
                scored_rules = [(r, score) for r, score in scored_rules if score > min_f1]
                scored_rules.sort(key=lambda x: x[1], reverse=True)
                balanced_rules = [r for r, _ in scored_rules[:max_rules]]
                
                if self.verbose:
                    print(f"  [Rule Filtering] Limited to top {len(balanced_rules)} rules by F1 score (from {len(all_rules)} total)")
                    if len(balanced_rules) > 0:
                        top_f1 = f1_score(balanced_rules[0])
                        print(f"  [Rule Filtering] Top rule F1 score: {top_f1:.4f}")
            else:
                # Even if we have fewer rules, filter out rules with very low F1
                # But be less strict - allow F1 > 0.05 (very low threshold)
                balanced_rules = [r for r in balanced_rules if f1_score(r) > 0.05]
                if self.verbose and len(balanced_rules) < len([r for r in all_rules if r.get('precision', 0) > min_precision]):
                    print(f"  [Rule Filtering] Filtered out {len([r for r in all_rules if r.get('precision', 0) > min_precision]) - len(balanced_rules)} rules with F1 <= 0.05")
            
            self.learned_rules = balanced_rules
            
            if self.verbose:
                print(f"\nLearning complete")
                print(f"Total unique verified rules: {len(all_rules)}")
                arithmetic_count = len([r for r in all_rules if r.get('type') == 'arithmetic_linear'])
                range_count = len([r for r in all_rules if r.get('type') == 'range'])
                other_count = len(all_rules) - arithmetic_count - range_count
                print(f"  - Arithmetic rules: {arithmetic_count}")
                print(f"  - Range rules: {range_count}")
                print(f"  - Other rules: {other_count}")
                print(f"\n[Rule Filtering] After filtering (precision > 0.6, recall >= 0.05): {len(self.learned_rules)} rules")
            print(f"\nTop 5 rules:")
            for i, rule in enumerate(self.learned_rules[:5], 1):
                    rule_type = rule.get('type', 'unknown')
                    print(f"{i}. Type: {rule_type}, Score: {rule.get('score', 0):.4f}, "
                      f"Precision: {rule.get('precision', 0):.4f}, "
                      f"Recall: {rule.get('recall', 0):.4f}")
                    if rule_type == 'arithmetic_linear':
                        features = rule.get('features', [])
                        coeffs = rule.get('coefficients', [])
                        threshold = rule.get('threshold', 0)
                        print(f"   {coeffs[0]}*{features[0]} + {coeffs[1]}*{features[1]} <= {threshold:.4f}")
                    elif rule_type == 'range':
                        feature = rule.get('feature', '')
                        lower = rule.get('lower_bound', 0)
                        upper = rule.get('upper_bound', 0)
                        print(f"   {lower:.4f} < {feature} < {upper:.4f}")
                    elif 'pygol_source' in rule:
                        print(f"   {rule['pygol_source']}")
            
            return self
        except Exception as e:
            # If fit fails completely, set empty rules
            if self.verbose:
                print(f"Error in fit(): {e}")
                import traceback
                traceback.print_exc()
            self.learned_rules = []
        return self
    
    def _is_degenerate_rule(self, rule: Dict) -> bool:
        """Check if a rule is degenerate (e.g., X =:= 0, Y =:= 0)"""
        if rule['type'] == 'conjunction':
            components = rule.get('components', [])
            # Check if all components are equality with threshold 0
            if len(components) >= 2:
                all_zero_equality = all(
                    comp.get('operation') == '==' and 
                    (comp.get('threshold', comp.get('value', 1)) == 0 or 
                     comp.get('threshold', comp.get('value', 1)) == 0.0)
                    for comp in components
                )
                if all_zero_equality:
                    return True
        elif rule['type'] == 'single_feature':
            # Filter single feature rules with == 0 that are likely degenerate
            if rule.get('operation') == '==' and (rule.get('threshold', rule.get('value', 1)) == 0 or rule.get('threshold', rule.get('value', 1)) == 0.0):
                # Only filter if it's not a meaningful rule (very low precision/coverage)
                if rule.get('precision', 1.0) < 0.1 or rule.get('pos_coverage', 1.0) < 0.1:
                    return True
        return False
    
    def _rule_signature(self, rule: Dict) -> Tuple:
        """Create signature for rule deduplication"""
        if rule['type'] == 'single_feature':
            return (rule['type'], rule['feature'], rule.get('operation'), 
                   rule.get('threshold', rule.get('value')))
        elif rule['type'] == 'arithmetic_linear':
            # For arithmetic rules, use features, coefficients, threshold, and operation
            features = tuple(sorted(rule.get('features', [])))
            coeffs = tuple(rule.get('coefficients', []))
            threshold = rule.get('threshold', 0)
            operation = rule.get('operation', '<=')
            return (rule['type'], features, coeffs, threshold, operation)
        elif rule['type'] == 'conjunction':
            comps = tuple(sorted([
                (c['feature'], c.get('operation'), c.get('threshold', c.get('value')))
                for c in rule.get('components', [])
            ]))
            return (rule['type'], comps)
        else:
            # Unknown type, use a generic signature
            return (rule.get('type', 'unknown'), str(rule))
    
    def _select_features_smt(self, X: pd.DataFrame, y: np.ndarray, max_features: Optional[int] = None) -> List[str]:
        """
        Phase 2.4: Feature selection via SMT
        Uses SMT to identify which features are most relevant for classification
        """
        if not self.use_optimization:
            return list(X.columns)  # Return all if optimization disabled
        
        if max_features is None:
            max_features = min(10, len(X.columns))  # Default: top 10 features
        
        opt = Optimize()
        feature_selected = {}
        
        # Create binary variables for each feature (1 = selected, 0 = not selected)
        for feature in X.columns:
            feat_var = Int(f'select_{feature}')
            opt.add(feat_var >= 0)
            opt.add(feat_var <= 1)
            feature_selected[feature] = feat_var
        
        # Constraint: select at most max_features
        selected_count = Sum([feature_selected[f] for f in X.columns])
        opt.add(selected_count <= max_features)
        opt.add(selected_count >= 1)  # At least one feature
        
        # Objective: maximize separation between positive and negative examples
        y_series = pd.Series(y, index=X.index)
        pos_indices = X.index[y_series == 1].tolist()
        neg_indices = X.index[y_series == 0].tolist()
        
        separation_scores = []
        
        # Sample examples for performance
        sample_size = min(20, len(pos_indices), len(neg_indices))
        for pos_idx in pos_indices[:sample_size]:
            for neg_idx in neg_indices[:sample_size]:
                # Calculate distance when using selected features
                distance_components = []
                for feature in X.columns:
                    feat_val_pos = Real(f'pos_{pos_idx}_{feature}')
                    feat_val_neg = Real(f'neg_{neg_idx}_{feature}')
                    val_pos = X.loc[pos_idx, feature]
                    val_neg = X.loc[neg_idx, feature]
                    # Try to convert to float, skip if it fails
                    try:
                        val_pos_float = float(pd.to_numeric(val_pos, errors='coerce'))
                        if not pd.isna(val_pos_float):
                            opt.add(feat_val_pos == val_pos_float)
                    except (TypeError, ValueError):
                        pass
                    try:
                        val_neg_float = float(pd.to_numeric(val_neg, errors='coerce'))
                        if not pd.isna(val_neg_float):
                            opt.add(feat_val_neg == val_neg_float)
                    except (TypeError, ValueError):
                        pass
                    
                    # Distance contribution (only if feature is selected)
                    diff = (feat_val_pos - feat_val_neg) ** 2
                    distance_components.append(If(feature_selected[feature] == 1, diff, 0))
                
                if distance_components:
                    total_distance = Sum(distance_components) if len(distance_components) > 1 else distance_components[0]
                    separation_scores.append(total_distance)
        
        # Maximize separation (distance between positive and negative examples)
        if separation_scores:
            total_separation = Sum(separation_scores) if len(separation_scores) > 1 else separation_scores[0]
            opt.maximize(total_separation)
        
        result = opt.check()
        if result == sat:
            model = opt.model()
            selected = []
            for feature in X.columns:
                try:
                    if model[feature_selected[feature]].as_long() == 1:
                        selected.append(feature)
                except:
                    pass
            return selected if selected else list(X.columns[:max_features])  # Fallback
        else:
            # Fallback: return top features by variance
            return list(X.columns[:max_features])
    
    def _fit_with_cross_validation(self, X: pd.DataFrame, y: np.ndarray):
        """
        Phase 2.3: Cross-validation integration with SMT
        Performs k-fold cross-validation with SMT optimization on each fold
        """
        if self.verbose:
            print(f"\n[Cross-Validation] Using {self.cv_folds}-fold CV with SMT optimization")
        
        # Preprocess data
        X_processed = self._preprocess_data(X)
        
        # Prepare examples for PyGOL
        Training_pos, Training_neg = self._prepare_pygol_examples(X_processed, y)
        
        # Generate bottom clauses
        constants = self._extract_constants(X_processed)
        bk_file_path = os.path.abspath(self.bk_file)
        pos_example_path = os.path.abspath("pos_example.f")
        neg_example_path = os.path.abspath("neg_example.n")
        
        try:
            P, N = bottom_clause_generation(
                file=bk_file_path,
                constant_set=constants,
                depth=2,
                container="dict",
                positive_example=pos_example_path,
                negative_example=neg_example_path,
                positive_file_dictionary="positive_bottom_clause",
                negative_file_dictionary="negative_bottom_clause"
            )
            
            # Create folds
            kfolds = pygol_folds(
                folds=self.cv_folds,
                shuffle=True,
                positive_file_dictionary=P,
                negative_file_dictionary=N
            )
            
            # Cross-validate with PyGOL
            cv_model = pygol_cross_validation(
                kfolds=kfolds,
                file=bk_file_path,
                k_fold=self.cv_folds,
                constant_set=constants,
                max_literals=self.max_literals,
                max_neg=0,
                min_pos=1,
                key_size=1,
                verbose=self.verbose
            )
            
            # Aggregate rules from all folds and apply SMT optimization
            all_fold_rules = []
            if hasattr(cv_model, 'hypothesis') and cv_model.hypothesis:
                # Convert PyGOL rules to internal format
                fold_rules = self._convert_pygol_rules(cv_model.hypothesis, X_processed)
                # Apply SMT optimization
                optimized_rules = self._verify_rules_with_smt(fold_rules, X_processed, y)
                all_fold_rules.extend(optimized_rules)
            
            # Store aggregated rules
            self.learned_rules = sorted(all_fold_rules, key=lambda x: x.get('score', 0.0), reverse=True)
            
            if self.verbose:
                print(f"\n[Cross-Validation] Aggregated {len(self.learned_rules)} rules from {self.cv_folds} folds")
            
        except Exception as e:
            if self.verbose:
                print(f"Cross-validation failed, falling back to standard fit: {e}")
            # Fallback to standard fit - recreate BK and run iterative
            if os.path.exists(self.bk_file):
                os.remove(self.bk_file)
            # Create initial background knowledge
            original_bk_file = getattr(self, 'original_bk_file', None)
            self._create_initial_background_knowledge(X_processed, y, original_bk_file=original_bk_file)
            
            pygol_rules = self._learn_rules_with_pygol(X_processed, y, 1)
            if pygol_rules:
                self.learned_rules = self._verify_rules_with_smt(pygol_rules, X_processed, y)
            else:
                self.learned_rules = []
        
        return self
    
    def predict(self, X: pd.DataFrame):
        """
        Make predictions using rule ensemble/voting methods
        Supports majority voting, weighted voting, and diversity-based voting
        """
        X_processed = self._preprocess_data(X)
        
        if self.ensemble_method == 'majority':
            return self._predict_majority_voting(X_processed)
        elif self.ensemble_method == 'weighted_voting':
            return self._predict_weighted_voting(X_processed)
        elif self.ensemble_method == 'diversity_voting':
            return self._predict_diversity_voting(X_processed)
        else:
            # Default: weighted voting
            return self._predict_weighted_voting(X_processed)
    
    def _predict_majority_voting(self, X: pd.DataFrame) -> np.ndarray:
        """Majority voting: each rule votes, majority wins"""
        predictions = np.zeros(len(X))
        
        for i in range(len(X)):
            sample = X.iloc[i]
            votes_1 = 0
            votes_0 = 0
            
            for rule in self.learned_rules:
                if self._apply_rule(rule, pd.DataFrame([sample])).iloc[0]:
                    votes_1 += 1
                else:
                    votes_0 += 1
            
            predictions[i] = 1 if votes_1 > votes_0 else 0
        
        return predictions
    
    def _predict_weighted_voting(self, X: pd.DataFrame) -> np.ndarray:
        """Weighted voting: rules vote with their scores as weights"""
        predictions = np.zeros(len(X))
        
        # For conjunction, interval3d, and multihalfplane problems, we need AND logic: require ALL components to match
        # Check if we have conjunction rules or if this is a conjunction/interval3d/multihalfplane problem
        has_conjunction_rules = any(r.get('type') == 'conjunction' for r in self.learned_rules)
        is_conjunction_problem = self.target_predicate == 'conjunction'
        is_interval3d_problem = self.target_predicate == 'interval3d'
        is_multihalfplane_problem = self.target_predicate == 'multihalfplane'
        
        # If this is a conjunction, interval3d, or multihalfplane problem, use AND logic: require multiple rule types to match
        if is_conjunction_problem or is_interval3d_problem or is_multihalfplane_problem or has_conjunction_rules:
            # Find arithmetic and range rules
            arithmetic_rules = [r for r in self.learned_rules if r.get('type') == 'arithmetic_linear']
            range_rules = [r for r in self.learned_rules if r.get('type') == 'range']
            conjunction_rules = [r for r in self.learned_rules if r.get('type') == 'conjunction']
            
            # For interval3d: always use relaxed bounds approach (even if conjunction rule exists)
            # This is because the strict conjunction rule doesn't generalize well
            if is_interval3d_problem and range_rules:
                x_ranges = [r for r in range_rules if r.get('feature') == 'x']
                y_ranges = [r for r in range_rules if r.get('feature') == 'y']
                z_ranges = [r for r in range_rules if r.get('feature') == 'z']
                
                if x_ranges and y_ranges and z_ranges:
                    # Use best range rule for each dimension
                    best_x = max(x_ranges, key=lambda r: r.get('score', 0))
                    best_y = max(y_ranges, key=lambda r: r.get('score', 0))
                    best_z = max(z_ranges, key=lambda r: r.get('score', 0))
                    
                
                    # This is more robust than learned ranges which may be too conservative
                    # Use preprocessed training data for consistency
                    if hasattr(self, 'X_train_processed_') and hasattr(self, 'y_train_'):
                        X_train_data = self.X_train_processed_
                        y_train_data = self.y_train_
                        
                        # Get positive examples
                        pos_mask = y_train_data == 1
                        if pos_mask.sum() > 0:
                            pos_X = X_train_data[pos_mask]
                            
                            
                            # Get learned bounds from best range rules
                            x_learned_lower = best_x.get('lower_bound', 0)
                            x_learned_upper = best_x.get('upper_bound', 0)
                            y_learned_lower = best_y.get('lower_bound', 0)
                            y_learned_upper = best_y.get('upper_bound', 0)
                            z_learned_lower = best_z.get('lower_bound', 0)
                            z_learned_upper = best_z.get('upper_bound', 0)
                            
                            
                            x_train_p10 = float(pos_X['x'].quantile(0.10))
                            x_train_p90 = float(pos_X['x'].quantile(0.90))
                            y_train_p10 = float(pos_X['y'].quantile(0.10))
                            y_train_p90 = float(pos_X['y'].quantile(0.90))
                            z_train_p10 = float(pos_X['z'].quantile(0.10))
                            z_train_p90 = float(pos_X['z'].quantile(0.90))
                            
                            # Also get min/max for union (to ensure we cover all training data)
                            x_train_min = float(pos_X['x'].min())
                            x_train_max = float(pos_X['x'].max())
                            y_train_min = float(pos_X['y'].min())
                            y_train_max = float(pos_X['y'].max())
                            z_train_min = float(pos_X['z'].min())
                            z_train_max = float(pos_X['z'].max())
                            
                            # Use the tighter of percentile bounds or min/max (whichever is more conservative)
                            # This balances robustness (percentiles) with coverage (min/max)
                            x_train_lower = min(x_train_p10, x_train_min)
                            x_train_upper = max(x_train_p90, x_train_max)
                            y_train_lower = min(y_train_p10, y_train_min)
                            y_train_upper = max(y_train_p90, y_train_max)
                            z_train_lower = min(z_train_p10, z_train_min)
                            z_train_upper = max(z_train_p90, z_train_max)
                            
                            # Use the union of learned bounds and training data bounds
                            # This ensures we cover both what was learned and what was seen in training
                            x_lower_base = min(x_learned_lower, x_train_lower)
                            x_upper_base = max(x_learned_upper, x_train_upper)
                            y_lower_base = min(y_learned_lower, y_train_lower)
                            y_upper_base = max(y_learned_upper, y_train_upper)
                            z_lower_base = min(z_learned_lower, z_train_lower)
                            z_upper_base = max(z_learned_upper, z_train_upper)
                            
                            # Calculate range and add generous margin for test data generalization
                            x_range = x_upper_base - x_lower_base
                            y_range = y_upper_base - y_lower_base
                            z_range = z_upper_base - z_lower_base
                            
                            
                            # For very small ranges (< 0.5), use 150% margin (reduced from 200%)
                            # For small ranges (< 1.0), use 100% margin (reduced from 150%)
                            # For larger ranges, use 75% margin (reduced from 100%)
                            # This tightens bounds slightly to improve precision while maintaining recall
                            x_margin_factor = 1.5 if x_range < 0.5 else (1.0 if x_range < 1.0 else 0.75)
                            y_margin_factor = 1.5 if y_range < 0.5 else (1.0 if y_range < 1.0 else 0.75)
                            z_margin_factor = 1.5 if z_range < 0.5 else (1.0 if z_range < 1.0 else 0.75)
                            
                            # Use slightly smaller minimum margin to tighten bounds
                            x_margin = max(x_range * x_margin_factor, 0.5)  # Reduced from 0.75
                            y_margin = max(y_range * y_margin_factor, 0.5)
                            z_margin = max(z_range * z_margin_factor, 0.5)
                            
                            # Expand bounds with generous margin
                            x_lower_relaxed = x_lower_base - x_margin
                            x_upper_relaxed = x_upper_base + x_margin
                            y_lower_relaxed = y_lower_base - y_margin
                            y_upper_relaxed = y_upper_base + y_margin
                            z_lower_relaxed = z_lower_base - z_margin
                            z_upper_relaxed = z_upper_base + z_margin
                        else:
                            # Fallback to learned ranges with margin
                            x_lower = best_x.get('lower_bound', 0)
                            x_upper = best_x.get('upper_bound', 0)
                            x_range_width = x_upper - x_lower
                            x_margin = max(x_range_width * 0.30, 0.2)
                            x_lower_relaxed = x_lower - x_margin
                            x_upper_relaxed = x_upper + x_margin
                            
                            y_lower = best_y.get('lower_bound', 0)
                            y_upper = best_y.get('upper_bound', 0)
                            y_range_width = y_upper - y_lower
                            y_margin = max(y_range_width * 0.30, 0.2)
                            y_lower_relaxed = y_lower - y_margin
                            y_upper_relaxed = y_upper + y_margin
                            
                            z_lower = best_z.get('lower_bound', 0)
                            z_upper = best_z.get('upper_bound', 0)
                            z_range_width = z_upper - z_lower
                            z_margin = max(z_range_width * 0.30, 0.2)
                            z_lower_relaxed = z_lower - z_margin
                            z_upper_relaxed = z_upper + z_margin
                    else:
                        # Fallback to learned ranges with margin if training data not available
                        x_lower = best_x.get('lower_bound', 0)
                        x_upper = best_x.get('upper_bound', 0)
                        x_range_width = x_upper - x_lower
                        x_margin = max(x_range_width * 0.30, 0.2)
                        x_lower_relaxed = x_lower - x_margin
                        x_upper_relaxed = x_upper + x_margin
                        
                        y_lower = best_y.get('lower_bound', 0)
                        y_upper = best_y.get('upper_bound', 0)
                        y_range_width = y_upper - y_lower
                        y_margin = max(y_range_width * 0.30, 0.2)
                        y_lower_relaxed = y_lower - y_margin
                        y_upper_relaxed = y_upper + y_margin
                        
                        z_lower = best_z.get('lower_bound', 0)
                        z_upper = best_z.get('upper_bound', 0)
                        z_range_width = z_upper - z_lower
                        z_margin = max(z_range_width * 0.30, 0.2)
                        z_lower_relaxed = z_lower - z_margin
                        z_upper_relaxed = z_upper + z_margin
                    
                    # Debug: Print bounds being used
                    if self.verbose and is_interval3d_problem:
                        print(f"[Interval3d Prediction] Using relaxed bounds:")
                        print(f"  X: {x_lower_relaxed:.4f} < x < {x_upper_relaxed:.4f}")
                        print(f"  Y: {y_lower_relaxed:.4f} < y < {y_upper_relaxed:.4f}")
                        print(f"  Z: {z_lower_relaxed:.4f} < z < {z_upper_relaxed:.4f}")
                    
                    for i in range(len(X)):
                        sample = X.iloc[i]
                        # Check if ALL 3 relaxed range rules match
                        x_val = pd.to_numeric(sample.get('x', 0), errors='coerce')
                        y_val = pd.to_numeric(sample.get('y', 0), errors='coerce')
                        z_val = pd.to_numeric(sample.get('z', 0), errors='coerce')
                        
                        if pd.isna(x_val) or pd.isna(y_val) or pd.isna(z_val):
                            predictions[i] = 0
                        else:
                            x_range_width = x_upper_relaxed - x_lower_relaxed
                            y_range_width = y_upper_relaxed - y_lower_relaxed
                            z_range_width = z_upper_relaxed - z_lower_relaxed
                            
                            # Distance from edges (normalized to [0, 1])
                            x_dist_from_lower = (x_val - x_lower_relaxed) / max(x_range_width, 0.001)
                            x_dist_from_upper = (x_upper_relaxed - x_val) / max(x_range_width, 0.001)
                            x_confidence = min(x_dist_from_lower, x_dist_from_upper)  # How far from nearest edge
                            
                            y_dist_from_lower = (y_val - y_lower_relaxed) / max(y_range_width, 0.001)
                            y_dist_from_upper = (y_upper_relaxed - y_val) / max(y_range_width, 0.001)
                            y_confidence = min(y_dist_from_lower, y_dist_from_upper)
                            
                            z_dist_from_lower = (z_val - z_lower_relaxed) / max(z_range_width, 0.001)
                            z_dist_from_upper = (z_upper_relaxed - z_val) / max(z_range_width, 0.001)
                            z_confidence = min(z_dist_from_lower, z_dist_from_upper)
                            
                            # Basic bounds check
                            x_match = (x_val > x_lower_relaxed) and (x_val < x_upper_relaxed)
                            y_match = (y_val > y_lower_relaxed) and (y_val < y_upper_relaxed)
                            z_match = (z_val > z_lower_relaxed) and (z_val < z_upper_relaxed)
                            
                            # For interval3d, require ALL 3 to match (strict AND)
                            # Additionally, require at least 2 dimensions to have confidence > 0.1 (not too close to edge)
                            # This reduces false positives from values right at the boundary
                            if x_match and y_match and z_match:
                                confident_dims = sum([x_confidence > 0.1, y_confidence > 0.1, z_confidence > 0.1])
                                # Require at least 2 dimensions to be confident (not at edge)
                                predictions[i] = 1 if confident_dims >= 2 else 0
                            else:
                                predictions[i] = 0
                else:
                    # Fallback to OR logic if we don't have all 3 range rules
                    for i in range(len(X)):
                        sample = X.iloc[i]
                        confidence_1 = 0.0
                        for rule in self.learned_rules:
                            rule_weight = rule.get('score', 0.0)
                            if self._apply_rule(rule, pd.DataFrame([sample])).iloc[0]:
                                confidence_1 += rule_weight
                        predictions[i] = 1 if confidence_1 > 0 else 0
            # If we have a combined conjunction rule, use it (it already has AND logic)
            # But only for non-interval3d problems (interval3d uses relaxed bounds above)
            elif conjunction_rules and not is_interval3d_problem:
                for i in range(len(X)):
                    sample = X.iloc[i]
                    # Use the best conjunction rule (highest score)
                    best_conjunction = max(conjunction_rules, key=lambda r: r.get('score', 0))
                    if self._apply_rule(best_conjunction, pd.DataFrame([sample])).iloc[0]:
                        predictions[i] = 1
                    else:
                        predictions[i] = 0
            # For conjunction: require BOTH arithmetic AND range rules to match
            elif is_conjunction_problem and arithmetic_rules and range_rules:
                for i in range(len(X)):
                    sample = X.iloc[i]
                    # Check if BOTH an arithmetic rule AND a range rule match
                    arithmetic_match = any(self._apply_rule(r, pd.DataFrame([sample])).iloc[0] for r in arithmetic_rules)
                    range_match = any(self._apply_rule(r, pd.DataFrame([sample])).iloc[0] for r in range_rules)
                    predictions[i] = 1 if (arithmetic_match and range_match) else 0
            # For multihalfplane: require at least 2 arithmetic rules to match (both constraints)
            elif is_multihalfplane_problem and len(arithmetic_rules) >= 2:
                # Multiple Halfplanes requires: (a1*x + b1*y <= c1) AND (a2*x + b2*y <= c2)
                # So we need at least 2 arithmetic rules to both match
                for i in range(len(X)):
                    sample = X.iloc[i]
                    # Count how many arithmetic rules match
                    matching_rules = sum(1 for r in arithmetic_rules if self._apply_rule(r, pd.DataFrame([sample])).iloc[0])
                    # Require at least 2 arithmetic rules to match (AND logic)
                    predictions[i] = 1 if matching_rules >= 2 else 0
            else:
                # Fallback to OR logic if we don't have both types
                for i in range(len(X)):
                    sample = X.iloc[i]
                    confidence_1 = 0.0
                    for rule in self.learned_rules:
                        rule_weight = rule.get('score', 0.0)
                        if self._apply_rule(rule, pd.DataFrame([sample])).iloc[0]:
                            confidence_1 += rule_weight
                    predictions[i] = 1 if confidence_1 > 0 else 0
        else:
            # For other problems, use OR logic (if ANY rule matches, positive)
            for i in range(len(X)):
                sample = X.iloc[i]
                confidence_1 = 0.0
                # Only matching rules vote positive
                # This is the correct interpretation: rules are OR-ed together (if ANY matches, positive)
                
                for rule in self.learned_rules:
                    rule_weight = rule.get('score', 0.0)
                    if self._apply_rule(rule, pd.DataFrame([sample])).iloc[0]:
                        confidence_1 += rule_weight
                
                # Predict positive if at least one rule matches (confidence_1 > 0)
                # This matches Prolog semantics: multiple rules are OR-ed together
                predictions[i] = 1 if confidence_1 > 0 else 0
        
        return predictions
    
    def _predict_diversity_voting(self, X: pd.DataFrame) -> np.ndarray:
        """
        Diversity-based voting: prioritize diverse rules
        Rules that cover different examples get higher weight
        """
        # Calculate rule diversity
        rule_diversity = self._calculate_rule_diversity(X)
        
        predictions = np.zeros(len(X))
        
        for i in range(len(X)):
            sample = X.iloc[i]
            confidence_1 = 0.0
            confidence_0 = 0.0
            
            for rule_idx, rule in enumerate(self.learned_rules):
                # Weight = score * diversity
                rule_weight = rule.get('score', 0.0) * rule_diversity.get(rule_idx, 1.0)
                if self._apply_rule(rule, pd.DataFrame([sample])).iloc[0]:
                    confidence_1 += rule_weight
                else:
                    confidence_0 += rule_weight
            
            predictions[i] = 1 if confidence_1 >= confidence_0 else 0
        
        return predictions
    
    def _calculate_rule_diversity(self, X: pd.DataFrame) -> Dict[int, float]:
        """
        Calculate diversity score for each rule
        Rules that cover different examples are more diverse
        """
        if not self.learned_rules:
            return {}
        
        # Calculate coverage for each rule
        rule_coverages = []
        for rule in self.learned_rules:
            mask = self._apply_rule(rule, X)
            rule_coverages.append(mask)
        
        # Calculate pairwise Jaccard distance (diversity)
        diversity_scores = {}
        for i, rule in enumerate(self.learned_rules):
            diversity_sum = 0.0
            for j, other_rule in enumerate(self.learned_rules):
                if i != j:
                    # Jaccard distance = 1 - Jaccard similarity
                    intersection = (rule_coverages[i] & rule_coverages[j]).sum()
                    union = (rule_coverages[i] | rule_coverages[j]).sum()
                    jaccard_sim = intersection / max(union, 1)
                    diversity_sum += (1 - jaccard_sim)
            
            # Average diversity with other rules
            diversity_scores[i] = diversity_sum / max(len(self.learned_rules) - 1, 1)
        
        return diversity_scores
    
    def predict_proba(self, X: pd.DataFrame):
        """
        Predict class probabilities using ensemble methods
        """
        X_processed = self._preprocess_data(X)
        
        probabilities = np.zeros((len(X_processed), 2))
        
        if self.ensemble_method == 'majority':
            # For majority voting, convert votes to probabilities
            for i in range(len(X_processed)):
                sample = X_processed.iloc[i]
                votes_1 = 0
                votes_0 = 0
                
                for rule in self.learned_rules:
                    if self._apply_rule(rule, pd.DataFrame([sample])).iloc[0]:
                        votes_1 += 1
                    else:
                        votes_0 += 1
                
                total_votes = votes_1 + votes_0
                if total_votes > 0:
                    probabilities[i, 1] = votes_1 / total_votes
                    probabilities[i, 0] = votes_0 / total_votes
                else:
                    probabilities[i, 0] = 0.5
                    probabilities[i, 1] = 0.5
        
        elif self.ensemble_method == 'diversity_voting':
            rule_diversity = self._calculate_rule_diversity(X_processed)
            for i in range(len(X_processed)):
                sample = X_processed.iloc[i]
                confidence_1 = 0.0
                confidence_0 = 0.0
                
                for rule_idx, rule in enumerate(self.learned_rules):
                    rule_weight = rule.get('score', 0.0) * rule_diversity.get(rule_idx, 1.0)
                    if self._apply_rule(rule, pd.DataFrame([sample])).iloc[0]:
                        confidence_1 += rule_weight
                    else:
                        confidence_0 += rule_weight
                
                total = confidence_1 + confidence_0
                if total > 0:
                    probabilities[i, 1] = confidence_1 / total
                    probabilities[i, 0] = confidence_0 / total
                else:
                    probabilities[i, 0] = 0.5
                    probabilities[i, 1] = 0.5
        
        else:  # weighted_voting (default)
            for i in range(len(X_processed)):
                sample = X_processed.iloc[i]
                confidence_1 = 0.0
                confidence_0 = 0.0
                
                for rule in self.learned_rules:
                    if self._apply_rule(rule, pd.DataFrame([sample])).iloc[0]:
                        confidence_1 += rule.get('score', 0.0)
                    else:
                        confidence_0 += rule.get('score', 0.0)
                
                total = confidence_1 + confidence_0
                if total > 0:
                    probabilities[i, 1] = confidence_1 / total
                    probabilities[i, 0] = confidence_0 / total
                else:
                    probabilities[i, 0] = 0.5
                    probabilities[i, 1] = 0.5
        
        return probabilities


if __name__ == "__main__":
    # Example usage
    from sklearn.model_selection import train_test_split
    from sklearn.datasets import make_classification
    
    # Create sample data
    X, y = make_classification(
        n_samples=100,
        n_features=5,
        n_informative=3,
        n_redundant=1,
        n_classes=2,
        random_state=42
    )
    
    X_df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_df, y, test_size=0.2, random_state=42
    )
    
    # Train model
    learner = IterativeSMTILPLearner(
        max_iterations=3,
        convergence_threshold=0.01,
        max_literals=3,
        verbose=True
    )
    
    learner.fit(X_train, y_train)
    
    # Evaluate
    y_pred = learner.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print("\nEvaluation results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

