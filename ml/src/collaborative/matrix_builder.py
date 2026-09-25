import pandas as pd
import numpy as np
from typing import Tuple, Dict, Optional

DEFAULT_WEIGHT_MAP = {
    'VIEW': 1.0,
    'BOOKMARK': 2.0,
    'ENROLL': 3.0,
    'COMPLETE': 4.0
}

def build_user_item_matrix(
    df_interactions: pd.DataFrame,
    weight_map: Optional[Dict[str, float]] = None
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Converts raw interaction events into a 2D User-Item interaction matrix.
    Aggregates implicit weights per (user_id, course_id).
    Unobserved entries are filled with 0.0.
    """
    if weight_map is None:
        weight_map = DEFAULT_WEIGHT_MAP
        
    df_imp = df_interactions[df_interactions['interaction_type'].isin(weight_map.keys())].copy()
    df_imp['weight'] = df_imp['interaction_type'].map(weight_map)
    
    summary = df_imp.groupby(['user_id', 'course_id'])['weight'].sum().reset_index()
    
    matrix = summary.pivot(
        index='user_id',
        columns='course_id',
        values='weight'
    ).fillna(0.0)
    
    n_users, n_courses = matrix.shape
    total_cells = n_users * n_courses
    observed_cells = (matrix > 0).sum().sum()
    density = (observed_cells / total_cells) * 100.0
    sparsity = 100.0 - density
    
    stats = {
        'n_users': n_users,
        'n_courses': n_courses,
        'total_cells': total_cells,
        'observed_cells': observed_cells,
        'density_pct': density,
        'sparsity_pct': sparsity
    }
    
    return matrix, stats
