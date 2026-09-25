import pytest
import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from content_based import (
    clean_text,
    build_combined_course_features,
    ContentBasedRecommender
)

def test_clean_text_formatting():
    """Verifies punctuation removal, lowercasing, and whitespace stripping."""
    raw = "  React, Redux & JavaScript - Masterclass!! "
    cleaned = clean_text(raw)
    assert cleaned == "react redux javascript masterclass"

def test_build_combined_course_features():
    """Verifies metadata feature string concatenation and repetitions."""
    df_courses = pd.DataFrame([{
        'course_id': 'C1',
        'title': 'Python Basics',
        'category': 'Programming',
        'skills': 'Python,OOP',
        'description': 'Learn Python programming.'
    }])
    
    features = build_combined_course_features(df_courses, title_weight=2, category_weight=1, skills_weight=1, description_weight=1)
    feat_text = features.iloc[0]
    
    assert feat_text.count("python basics") == 2
    assert "programming" in feat_text
    assert "python oop" in feat_text

def test_recommend_similar_courses_ranking_and_exclusion():
    """Verifies item-to-item similarity ranking and source course exclusion."""
    df_courses = pd.DataFrame([
        {'course_id': 'C1', 'title': 'Python Data Science', 'category': 'Data', 'skills': 'Python,Pandas', 'description': 'Learn Pandas.'},
        {'course_id': 'C2', 'title': 'Python Machine Learning', 'category': 'Data', 'skills': 'Python,Scikit-Learn', 'description': 'Learn ML.'},
        {'course_id': 'C3', 'title': 'Figma UI Design', 'category': 'Design', 'skills': 'Figma,UX', 'description': 'Learn Design.'},
    ])
    df_interactions = pd.DataFrame([
        {'user_id': 'U1', 'course_id': 'C1', 'interaction_type': 'VIEW'},
    ])
    
    recommender = ContentBasedRecommender()
    recommender.fit(df_courses, df_interactions)
    
    similar_recs = recommender.recommend_similar_courses('C1', top_n=2)
    rec_ids = similar_recs['course_id'].tolist()
    
    assert 'C1' not in rec_ids
    assert rec_ids[0] == 'C2'

def test_personalized_recommendation_filtering_and_fallback():
    """Verifies candidate filtering of interacted courses and cold-start fallback."""
    df_courses = pd.DataFrame([
        {'course_id': 'C1', 'title': 'Python Data Science', 'category': 'Data', 'skills': 'Python,Pandas', 'description': 'Learn Pandas.'},
        {'course_id': 'C2', 'title': 'Python Machine Learning', 'category': 'Data', 'skills': 'Python,Scikit-Learn', 'description': 'Learn ML.'},
        {'course_id': 'C3', 'title': 'Figma UI Design', 'category': 'Design', 'skills': 'Figma,UX', 'description': 'Learn Design.'},
    ])
    df_interactions = pd.DataFrame([
        {'user_id': 'U1', 'course_id': 'C1', 'interaction_type': 'ENROLL'},
    ])
    
    recommender = ContentBasedRecommender()
    recommender.fit(df_courses, df_interactions)
    
    u1_recs = recommender.recommend('U1', top_n=2)
    assert 'C1' not in u1_recs['course_id'].tolist()
    assert u1_recs['course_id'].iloc[0] == 'C2'
    assert u1_recs['recommendation_type'].iloc[0] == 'content_based'
    
    u99_recs = recommender.recommend('U99', top_n=1)
    assert u99_recs['recommendation_type'].iloc[0] == 'popularity_fallback'
