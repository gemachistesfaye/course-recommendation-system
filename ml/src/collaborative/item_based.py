import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
from sklearn.metrics.pairwise import cosine_similarity

from .matrix_builder import build_user_item_matrix
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from preprocessing.recommender import PopularityRecommender

class ItemBasedRecommender:
    """
    Item-Item Collaborative Filtering Recommender based on user-item interaction
    column similarities and user-history weighted candidate aggregation.
    """
    
    def __init__(self, top_k_similar: int = 10):
        self.top_k_similar = top_k_similar
        self.user_item_matrix: pd.DataFrame = None
        self.item_sim_matrix: np.ndarray = None
        self.user_ids: List[str] = []
        self.course_ids: List[str] = []
        self.course_to_idx: Dict[str, int] = {}
        self.user_to_idx: Dict[str, int] = {}
        self.courses_df: pd.DataFrame = None
        self.popularity_fallback: PopularityRecommender = None

    def fit(self, df_interactions: pd.DataFrame, courses_df: pd.DataFrame) -> 'ItemBasedRecommender':
        """Fits user-item interaction matrix and item-item similarity matrix."""
        self.courses_df = courses_df.copy()
        self.user_item_matrix, _ = build_user_item_matrix(df_interactions)
        
        self.user_ids = list(self.user_item_matrix.index)
        self.course_ids = list(self.user_item_matrix.columns)
        self.course_to_idx = {cid: idx for idx, cid in enumerate(self.course_ids)}
        self.user_to_idx = {uid: idx for idx, uid in enumerate(self.user_ids)}
        
        self.item_sim_matrix = cosine_similarity(self.user_item_matrix.values.T)
        np.fill_diagonal(self.item_sim_matrix, 0.0)
        
        df_imp = df_interactions[df_interactions['interaction_type'] != 'RATE'].copy()
        weight_map = {'VIEW': 1.0, 'BOOKMARK': 2.0, 'ENROLL': 3.0, 'COMPLETE': 4.0}
        df_imp['weight'] = df_imp['interaction_type'].map(weight_map)
        summary = df_imp.groupby(['user_id', 'course_id'])['weight'].sum().reset_index()
        
        self.popularity_fallback = PopularityRecommender()
        self.popularity_fallback.fit(summary, self.courses_df)
        
        return self

    def recommend_collaborative_similar_courses(self, course_id: str, top_n: int = 5) -> pd.DataFrame:
        """Item-to-Item Collaborative recommendation: returns top_n courses co-consumed with course_id."""
        if course_id not in self.course_to_idx:
            raise ValueError(f"Course ID '{course_id}' not found in interaction matrix.")
            
        c_idx = self.course_to_idx[course_id]
        item_sims = self.item_sim_matrix[c_idx]
        
        sim_df = pd.DataFrame({
            'course_id': self.course_ids,
            'similarity_score': item_sims
        })
        
        sim_df = sim_df[sim_df['course_id'] != course_id]
        top_candidates = sim_df.sort_values(by='similarity_score', ascending=False).head(top_n)
        
        result = top_candidates.merge(self.courses_df, on='course_id', how='left')
        result['recommendation_type'] = 'item_collaborative_similar'
        return result

    def recommend(self, user_id: str, top_n: int = 5) -> pd.DataFrame:
        """Personalized Item-Item Collaborative recommendation for user_id."""
        if user_id not in self.user_to_idx:
            fallback = self.popularity_fallback.recommend(user_id, top_n=top_n)
            fallback['recommendation_type'] = 'popularity_fallback'
            return fallback
            
        u_idx = self.user_to_idx[user_id]
        user_vector = self.user_item_matrix.iloc[u_idx].values
        
        interacted_indices = np.where(user_vector > 0)[0]
        if len(interacted_indices) == 0:
            fallback = self.popularity_fallback.recommend(user_id, top_n=top_n)
            fallback['recommendation_type'] = 'popularity_fallback'
            return fallback
            
        candidate_scores = {}
        
        for c_idx, cid in enumerate(self.course_ids):
            if user_vector[c_idx] > 0:
                continue
                
            score_num = 0.0
            score_den = 0.0
            
            for i_idx in interacted_indices:
                w_user = user_vector[i_idx]
                sim = self.item_sim_matrix[c_idx, i_idx]
                if sim > 0:
                    score_num += sim * w_user
                    score_den += sim
                    
            if score_den > 0:
                candidate_scores[cid] = score_num / score_den
                
        if not candidate_scores:
            fallback = self.popularity_fallback.recommend(user_id, top_n=top_n)
            fallback['recommendation_type'] = 'popularity_fallback'
            return fallback
            
        rec_df = pd.DataFrame(list(candidate_scores.items()), columns=['course_id', 'predicted_score'])
        rec_df = rec_df.sort_values(by='predicted_score', ascending=False).head(top_n)
        
        result = rec_df.merge(self.courses_df, on='course_id', how='left')
        result['recommendation_type'] = 'item_item_collaborative'
        return result
