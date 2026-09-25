import pandas as pd
import numpy as np
from typing import Dict, Tuple

DEFAULT_WEIGHT_MAP = {
    'VIEW': 1.0,
    'BOOKMARK': 2.0,
    'ENROLL': 3.0,
    'COMPLETE': 4.0
}

REQUIRED_USER_COLS = {'user_id'}
REQUIRED_COURSE_COLS = {'course_id', 'title', 'category'}
REQUIRED_INTERACTION_COLS = {'user_id', 'course_id', 'interaction_type'}

def validate_datasets(df_users: pd.DataFrame, df_courses: pd.DataFrame, df_interactions: pd.DataFrame) -> None:
    """Validates that all required columns are present in datasets."""
    if not REQUIRED_USER_COLS.issubset(df_users.columns):
        missing = REQUIRED_USER_COLS - set(df_users.columns)
        raise ValueError(f"df_users missing required columns: {missing}")
        
    if not REQUIRED_COURSE_COLS.issubset(df_courses.columns):
        missing = REQUIRED_COURSE_COLS - set(df_courses.columns)
        raise ValueError(f"df_courses missing required columns: {missing}")
        
    if not REQUIRED_INTERACTION_COLS.issubset(df_interactions.columns):
        missing = REQUIRED_INTERACTION_COLS - set(df_interactions.columns)
        raise ValueError(f"df_interactions missing required columns: {missing}")

def clean_interactions(df_interactions: pd.DataFrame, valid_types=None) -> pd.DataFrame:
    """Removes invalid interaction types, drops NaNs in IDs, and eliminates duplicates."""
    if valid_types is None:
        valid_types = {'VIEW', 'BOOKMARK', 'ENROLL', 'COMPLETE', 'RATE'}
        
    df_clean = df_interactions.dropna(subset=['user_id', 'course_id', 'interaction_type']).copy()
    df_clean = df_clean[df_clean['interaction_type'].isin(valid_types)]
    df_clean = df_clean.drop_duplicates()
    return df_clean

def compute_interaction_weights(
    df_interactions: pd.DataFrame, 
    weight_map: Dict[str, float] = None
) -> pd.DataFrame:
    """
    Converts implicit interaction events to numerical score weights.
    Aggregates weights per (user_id, course_id).
    """
    if weight_map is None:
        weight_map = DEFAULT_WEIGHT_MAP
        
    # Exclude explicit RATE interactions for implicit aggregation
    df_imp = df_interactions[df_interactions['interaction_type'].isin(weight_map.keys())].copy()
    df_imp['weight'] = df_imp['interaction_type'].map(weight_map)
    
    summary = df_imp.groupby(['user_id', 'course_id'])['weight'].sum().reset_index()
    return summary

def create_user_item_matrix(interaction_summary: pd.DataFrame) -> pd.DataFrame:
    """Pivots interaction summary into a 2D User-Item matrix."""
    matrix = interaction_summary.pivot(
        index='user_id',
        columns='course_id',
        values='weight'
    ).fillna(0.0)
    return matrix
