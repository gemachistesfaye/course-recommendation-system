import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
from sklearn.metrics.pairwise import cosine_similarity

from .matrix_builder import build_user_item_matrix
import sys
import os

# Import PopularityRecommender fallback from preprocessing
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from preprocessing.recommender import PopularityRecommender

class UserBasedRecommender:
    """
    User-User Collaborative Filtering Recommender based on user vector
    cosine similarities and weighted neighbor candidate aggregation.
    """
    
    def __init__(self, k_neighbors: int = 10):
        self.k_neighbors = k_neighbors
        self.user_item_matrix: pd.DataFrame = None
        self.user_sim_matrix: np.ndarray = None
        self.user_ids: List[str] = []
        self.course_ids: List[str] = []
        self.user_to_idx: Dict[str, int] = {}
        self.courses_df: pd.DataFrame = None
        self.popularity_fallback: PopularityRecommender = None

    def fit(self, df_interactions: pd.DataFrame, courses_df: pd.DataFrame) -> 'UserBasedRecommender':
        """Fits user-item interaction matrix and user similarity matrix."""
        self.courses_df = courses_df.copy()
        self.user_item_matrix, _ = build_user_item_matrix(df_interactions)
        
        self.user_ids = list(self.user_item_matrix.index)
        self.course_ids = list(self.user_item_matrix.columns)
        self.user_to_idx = {uid: idx for idx, uid in enumerate(self.user_ids)}
        
        # Compute 50 x 50 User-User Cosine Similarity Matrix
        self.user_sim_matrix = cosine_similarity(self.user_item_matrix.values)
        
        # Zero out self-similarity on diagonal to avoid self-neighboring
        np.fill_diagonal(self.user_sim_matrix, 0.0)
        
        # Fit Popularity fallback for cold-start users
        df_imp = df_interactions[df_interactions['interaction_type'] != 'RATE'].copy()
        weight_map = {'VIEW': 1.0, 'BOOKMARK': 2.0, 'ENROLL': 3.0, 'COMPLETE': 4.0}
        df_imp['weight'] = df_imp['interaction_type'].map(weight_map)
        summary = df_imp.groupby(['user_id', 'course_id'])['weight'].sum().reset_index()
        
        self.popularity_fallback = PopularityRecommender()
        self.popularity_fallback.fit(summary, self.courses_df)
        
        return self

    def recommend(self, user_id: str, top_n: int = 5) -> pd.DataFrame:
        """Generates Top-N recommendations for user_id based on K-nearest similar users."""
        if user_id not in self.user_to_idx:
            fallback = self.popularity_fallback.recommend(user_id, top_n=top_n)
            fallback['recommendation_type'] = 'popularity_fallback'
            return fallback
            
        u_idx = self.user_to_idx[user_id]
        user_sims = self.user_sim_matrix[u_idx]
        
        # Check if user has zero similarity with all other users
        if np.max(user_sims) <= 0:
            fallback = self.popularity_fallback.recommend(user_id, top_n=top_n)
            fallback['recommendation_type'] = 'popularity_fallback'
            return fallback
            
        # Get Top-K nearest neighbors indices
        neighbor_indices = np.argsort(user_sims)[::-1][:self.k_neighbors]
        # Keep only positive similarities
        neighbor_indices = [idx for idx in neighbor_indices if user_sims[idx] > 0]
        
        if not neighbor_indices:
            fallback = self.popularity_fallback.recommend(user_id, top_n=top_n)
            fallback['recommendation_type'] = 'popularity_fallback'
            return fallback
            
        # Extract target user's interaction vector
        target_user_vector = self.user_item_matrix.iloc[u_idx].values
        
        # Compute predicted scores for candidate courses
        candidate_scores = {}
        
        for c_idx, cid in enumerate(self.course_ids):
            # Exclude courses user has already interacted with
            if target_user_vector[c_idx] > 0:
                continue
                
            score_num = 0.0
            score_den = 0.0
            
            for n_idx in neighbor_indices:
                sim = user_sims[n_idx]
                r_val = self.user_item_matrix.iloc[n_idx, c_idx]
                if r_val > 0:
                    score_num += sim * r_val
                    score_den += sim
                    
            if score_den > 0:
                candidate_scores[cid] = score_num / score_den
                
        if not candidate_scores:
            fallback = self.popularity_fallback.recommend(user_id, top_n=top_n)
            fallback['recommendation_type'] = 'popularity_fallback'
            return fallback
            
        # Convert to DataFrame, sort and merge course metadata
        rec_df = pd.DataFrame(list(candidate_scores.items()), columns=['course_id', 'predicted_score'])
        rec_df = rec_df.sort_values(by='predicted_score', ascending=False).head(top_n)
        
        result = rec_df.merge(self.courses_df, on='course_id', how='left')
        result['recommendation_type'] = 'user_user_collaborative'
        return result
