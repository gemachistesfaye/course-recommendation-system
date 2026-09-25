import pytest
import pandas as pd
import sys
import os

# Include src directory in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from preprocessing import (
    validate_datasets,
    clean_interactions,
    compute_interaction_weights,
    PopularityRecommender
)

def test_clean_interactions_filters_invalid_types_and_nans():
    """Verifies filtering of invalid interaction types, NaNs, and duplicate records."""
    raw_data = pd.DataFrame([
        {'user_id': 'U1', 'course_id': 'C1', 'interaction_type': 'VIEW'},
        {'user_id': 'U1', 'course_id': 'C2', 'interaction_type': 'INVALID_ACTION'},
        {'user_id': None, 'course_id': 'C3', 'interaction_type': 'ENROLL'},
        {'user_id': 'U2', 'course_id': 'C1', 'interaction_type': 'VIEW'},
        {'user_id': 'U2', 'course_id': 'C1', 'interaction_type': 'VIEW'}, # Duplicate
    ])
    
    clean_df = clean_interactions(raw_data)
    assert len(clean_df) == 2  # Only valid U1-C1 VIEW and U2-C1 VIEW kept
    assert 'INVALID_ACTION' not in clean_df['interaction_type'].values

def test_validate_datasets_missing_columns():
    """Verifies schema validation when required columns are missing."""
    df_users = pd.DataFrame([{'name': 'Alice'}]) # Missing user_id
    df_courses = pd.DataFrame([{'course_id': 'C1', 'title': 'Course 1', 'category': 'Web'}])
    df_interactions = pd.DataFrame([{'user_id': 'U1', 'course_id': 'C1', 'interaction_type': 'VIEW'}])
    
    with pytest.raises(ValueError, match="df_users missing required columns"):
        validate_datasets(df_users, df_courses, df_interactions)

def test_compute_interaction_weights():
    """Verifies weight mapping conversion for implicit feedback."""
    interactions = pd.DataFrame([
        {'user_id': 'U1', 'course_id': 'C1', 'interaction_type': 'VIEW'},
        {'user_id': 'U1', 'course_id': 'C1', 'interaction_type': 'ENROLL'},
        {'user_id': 'U1', 'course_id': 'C2', 'interaction_type': 'BOOKMARK'},
    ])
    
    weights_df = compute_interaction_weights(interactions)
    u1_c1_weight = weights_df[(weights_df['user_id'] == 'U1') & (weights_df['course_id'] == 'C1')]['weight'].values[0]
    
    # VIEW (1) + ENROLL (3) = 4
    assert u1_c1_weight == 4.0

def test_popularity_recommender_filtering_and_ranking():
    """Verifies global candidate ranking and exclusion of already-interacted courses."""
    summary = pd.DataFrame([
        {'user_id': 'U1', 'course_id': 'C1', 'weight': 5.0},
        {'user_id': 'U1', 'course_id': 'C2', 'weight': 3.0},
        {'user_id': 'U2', 'course_id': 'C1', 'weight': 5.0},
        {'user_id': 'U2', 'course_id': 'C3', 'weight': 10.0},
        {'user_id': 'U3', 'course_id': 'C4', 'weight': 8.0},
    ])
    courses = pd.DataFrame([
        {'course_id': 'C1', 'title': 'Course 1', 'category': 'Web'},
        {'course_id': 'C2', 'title': 'Course 2', 'category': 'Web'},
        {'course_id': 'C3', 'title': 'Course 3', 'category': 'Data'},
        {'course_id': 'C4', 'title': 'Course 4', 'category': 'AI'},
    ])
    
    recommender = PopularityRecommender()
    recommender.fit(summary, courses)
    
    # Global popularity totals: C1: 10, C3: 10, C4: 8, C2: 3
    # U1 already took C1 and C2 -> Should recommend C3 first, then C4
    recs = recommender.recommend('U1', top_n=2)
    rec_course_ids = recs['course_id'].tolist()
    
    assert rec_course_ids == ['C3', 'C4']
    assert 'C1' not in rec_course_ids
    assert 'C2' not in rec_course_ids

def test_popularity_recommender_cold_start():
    """Verifies that unknown/new users receive overall top popular courses."""
    summary = pd.DataFrame([
        {'user_id': 'U1', 'course_id': 'C1', 'weight': 10.0},
        {'user_id': 'U1', 'course_id': 'C2', 'weight': 5.0},
    ])
    courses = pd.DataFrame([
        {'course_id': 'C1', 'title': 'Course 1', 'category': 'Web'},
        {'course_id': 'C2', 'title': 'Course 2', 'category': 'Web'},
    ])
    
    recommender = PopularityRecommender()
    recommender.fit(summary, courses)
    
    # Unknown user 'U99' should get top popular course 'C1'
    recs = recommender.recommend('U99', top_n=1)
    assert recs['course_id'].iloc[0] == 'C1'
