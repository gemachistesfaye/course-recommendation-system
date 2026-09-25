import pytest
import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from collaborative import (
    build_user_item_matrix,
    UserBasedRecommender,
    ItemBasedRecommender,
    MatrixFactorizationRecommender
)

@pytest.fixture
def sample_data():
    df_courses = pd.DataFrame([
        {'course_id': 'C1', 'title': 'Python Basics', 'category': 'Programming'},
        {'course_id': 'C2', 'title': 'Pandas Data Analysis', 'category': 'Data'},
        {'course_id': 'C3', 'title': 'Docker DevOps', 'category': 'DevOps'},
        {'course_id': 'C4', 'title': 'React Fundamentals', 'category': 'Web'},
    ])
    df_interactions = pd.DataFrame([
        {'user_id': 'U1', 'course_id': 'C1', 'interaction_type': 'VIEW'},
        {'user_id': 'U1', 'course_id': 'C2', 'interaction_type': 'ENROLL'},
        {'user_id': 'U2', 'course_id': 'C1', 'interaction_type': 'VIEW'},
        {'user_id': 'U2', 'course_id': 'C2', 'interaction_type': 'COMPLETE'},
        {'user_id': 'U2', 'course_id': 'C3', 'interaction_type': 'ENROLL'},
        {'user_id': 'U3', 'course_id': 'C4', 'interaction_type': 'ENROLL'},
    ])
    return df_courses, df_interactions

def test_build_user_item_matrix(sample_data):
    df_courses, df_interactions = sample_data
    matrix, stats = build_user_item_matrix(df_interactions)
    
    assert matrix.shape == (3, 4)
    assert stats['n_users'] == 3
    assert stats['n_courses'] == 4
    # U1 on C2 weight = ENROLL (3.0)
    assert matrix.loc['U1', 'C2'] == 3.0

def test_user_based_recommender(sample_data):
    df_courses, df_interactions = sample_data
    model = UserBasedRecommender(k_neighbors=2).fit(df_interactions, df_courses)
    
    # U1 took C1, C2. Similar to U2 (took C1, C2, C3) -> Should recommend C3 to U1
    recs = model.recommend('U1', top_n=2)
    rec_ids = recs['course_id'].tolist()
    
    assert 'C1' not in rec_ids
    assert 'C2' not in rec_ids
    assert rec_ids[0] == 'C3'

def test_item_based_recommender_exclusion_and_ranking(sample_data):
    df_courses, df_interactions = sample_data
    model = ItemBasedRecommender().fit(df_interactions, df_courses)
    
    # Item similarity for C1 should exclude C1
    similar_df = model.recommend_collaborative_similar_courses('C1', top_n=2)
    assert 'C1' not in similar_df['course_id'].tolist()
    
    # Personal recommendation for U1 should exclude C1 and C2
    recs = model.recommend('U1', top_n=2)
    assert 'C1' not in recs['course_id'].tolist()
    assert 'C2' not in recs['course_id'].tolist()

def test_matrix_factorization_recommender(sample_data):
    df_courses, df_interactions = sample_data
    model = MatrixFactorizationRecommender(n_components=2).fit(df_interactions, df_courses)
    
    recs = model.recommend('U1', top_n=2)
    assert len(recs) == 2
    assert 'C1' not in recs['course_id'].tolist()
    assert 'C2' not in recs['course_id'].tolist()

def test_collaborative_cold_start_fallback(sample_data):
    df_courses, df_interactions = sample_data
    model = UserBasedRecommender().fit(df_interactions, df_courses)
    
    # Unknown user U99 should fall back to popularity
    recs = model.recommend('U99', top_n=2)
    assert recs['recommendation_type'].iloc[0] == 'popularity_fallback'
