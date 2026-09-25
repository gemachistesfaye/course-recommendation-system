import pandas as pd
import numpy as np
from typing import List, Dict

class PopularityRecommender:
    """Simple baseline recommender based on global course popularity scores."""
    
    def __init__(self):
        self.popularity_scores: pd.Series = None
        self.user_history: Dict[str, set] = {}
        self.courses_df: pd.DataFrame = None

    def fit(self, interactions_summary: pd.DataFrame, courses_df: pd.DataFrame) -> None:
        """
        Fits the baseline model by calculating global course popularity scores
        and storing user interaction histories.
        """
        self.courses_df = courses_df.set_index('course_id') if 'course_id' in courses_df.columns else courses_df
        
        # Calculate total interaction score per course
        self.popularity_scores = interactions_summary.groupby('course_id')['weight'].sum().sort_values(ascending=False)
        
        # Store set of interacted courses per user
        user_grouped = interactions_summary.groupby('user_id')['course_id'].apply(set)
        self.user_history = user_grouped.to_dict()

    def recommend(self, user_id: str, top_n: int = 5) -> pd.DataFrame:
        """
        Returns top_n course recommendations for a user, excluding previously interacted courses.
        If user_id is unknown (cold start), returns overall top courses.
        """
        interacted_courses = self.user_history.get(user_id, set())
        
        # Filter out already interacted courses
        candidate_scores = self.popularity_scores[~self.popularity_scores.index.isin(interacted_courses)]
        
        top_candidates = candidate_scores.head(top_n).reset_index()
        top_candidates.columns = ['course_id', 'popularity_score']
        
        # Join with course details for rich output
        if self.courses_df is not None:
            result = top_candidates.merge(self.courses_df, on='course_id', how='left')
        else:
            result = top_candidates
            
        return result
