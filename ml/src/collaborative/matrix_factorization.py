import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
from sklearn.decomposition import TruncatedSVD

from .matrix_builder import build_user_item_matrix
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from preprocessing.recommender import PopularityRecommender

class MatrixFactorizationRecommender:
    """
    Collaborative Filtering Recommender using Low-Rank Matrix Factorization
    via Truncated Singular Value Decomposition (SVD).
    """
    
    def __init__(self, n_components: int = 5, random_state: int = 42):
        self.n_components = n_components
        self.random_state = random_state
        self.svd = TruncatedSVD(n_components=n_components, random_state=random_state)
        self.user_item_matrix: pd.DataFrame = None
        self.reconstructed_matrix: pd.DataFrame = None
        self.user_ids: List[str] = []
        self.course_ids: List[str] = []
        self.user_to_idx: Dict[str, int] = {}
        self.courses_df: pd.DataFrame = None
        self.popularity_fallback: PopularityRecommender = None

    def fit(self, df_interactions: pd.DataFrame, courses_df: pd.DataFrame) -> 'MatrixFactorizationRecommender':
        """Fits Truncated SVD matrix factorization on the User-Item interaction matrix."""
        self.courses_df = courses_df.copy()
        self.user_item_matrix, _ = build_user_item_matrix(df_interactions)
        
        self.user_ids = list(self.user_item_matrix.index)
        self.course_ids = list(self.user_item_matrix.columns)
        self.user_to_idx = {uid: idx for idx, uid in enumerate(self.user_ids)}
        
        user_latent_P = self.svd.fit_transform(self.user_item_matrix.values)
        item_latent_Q = self.svd.components_
        
        pred_array = np.dot(user_latent_P, item_latent_Q)
        self.reconstructed_matrix = pd.DataFrame(
            pred_array,
            index=self.user_ids,
            columns=self.course_ids
        )
        
        df_imp = df_interactions[df_interactions['interaction_type'] != 'RATE'].copy()
        weight_map = {'VIEW': 1.0, 'BOOKMARK': 2.0, 'ENROLL': 3.0, 'COMPLETE': 4.0}
        df_imp['weight'] = df_imp['interaction_type'].map(weight_map)
        summary = df_imp.groupby(['user_id', 'course_id'])['weight'].sum().reset_index()
        
        self.popularity_fallback = PopularityRecommender()
        self.popularity_fallback.fit(summary, self.courses_df)
        
        return self

    def recommend(self, user_id: str, top_n: int = 5) -> pd.DataFrame:
        """Personalized Matrix Factorization recommendation for user_id."""
        if user_id not in self.user_to_idx:
            fallback = self.popularity_fallback.recommend(user_id, top_n=top_n)
            fallback['recommendation_type'] = 'popularity_fallback'
            return fallback
            
        u_idx = self.user_to_idx[user_id]
        observed_vector = self.user_item_matrix.iloc[u_idx]
        predicted_vector = self.reconstructed_matrix.iloc[u_idx].copy()
        
        interacted_courses = observed_vector[observed_vector > 0].index
        predicted_vector = predicted_vector[~predicted_vector.index.isin(interacted_courses)]
        
        top_candidates = predicted_vector.sort_values(ascending=False).head(top_n).reset_index()
        top_candidates.columns = ['course_id', 'predicted_score']
        
        result = top_candidates.merge(self.courses_df, on='course_id', how='left')
        result['recommendation_type'] = 'matrix_factorization_svd'
        return result
