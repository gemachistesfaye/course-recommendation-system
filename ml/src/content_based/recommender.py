import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from content_based.feature_builder import build_combined_course_features
from preprocessing.recommender import PopularityRecommender
from preprocessing.data_loader import compute_interaction_weights

class ContentBasedRecommender:
    """
    Personalized Content-Based Recommender using TF-IDF feature vectors,
    weighted user profile aggregation, and cosine similarity scoring.
    """
    
    def __init__(self, ngram_range: Tuple[int, int] = (1, 2), stop_words: str = 'english'):
        self.vectorizer = TfidfVectorizer(ngram_range=ngram_range, stop_words=stop_words)
        self.courses_df: pd.DataFrame = None
        self.tfidf_matrix: np.ndarray = None
        self.course_id_to_idx: Dict[str, int] = {}
        self.user_history: Dict[str, set] = {}
        self.user_interaction_weights: pd.DataFrame = None
        self.popularity_fallback: PopularityRecommender = None

    def fit(
        self, 
        courses_df: pd.DataFrame, 
        interactions_df: pd.DataFrame,
        weight_map: Optional[Dict[str, float]] = None
    ) -> 'ContentBasedRecommender':
        """Fits TF-IDF model, computes course vectors, and stores user interaction histories."""
        self.courses_df = courses_df.copy()
        
        # Build course feature strings & TF-IDF matrix
        combined_features = build_combined_course_features(self.courses_df)
        self.tfidf_matrix = self.vectorizer.fit_transform(combined_features)
        
        # Build course_id index lookup
        for idx, cid in enumerate(self.courses_df['course_id']):
            self.course_id_to_idx[cid] = idx
            
        # Compute user interaction weights for profile building
        self.user_interaction_weights = compute_interaction_weights(interactions_df, weight_map=weight_map)
        
        # Store user history sets for candidate filtering
        user_grouped = interactions_df.groupby('user_id')['course_id'].apply(set)
        self.user_history = user_grouped.to_dict()
        
        # Fit PopularityRecommender for cold-start fallback
        self.popularity_fallback = PopularityRecommender()
        self.popularity_fallback.fit(self.user_interaction_weights, self.courses_df)
        
        return self

    def recommend_similar_courses(self, course_id: str, top_n: int = 5) -> pd.DataFrame:
        """Item-to-Item recommendation: returns top_n courses similar to source course_id."""
        if course_id not in self.course_id_to_idx:
            raise ValueError(f"Course ID '{course_id}' not found in course catalog.")
            
        source_idx = self.course_id_to_idx[course_id]
        source_vector = self.tfidf_matrix[source_idx]
        
        # Compute similarity against all courses
        sim_scores = cosine_similarity(source_vector, self.tfidf_matrix).flatten()
        
        sim_df = pd.DataFrame({
            'course_id': self.courses_df['course_id'],
            'similarity_score': sim_scores
        })
        
        # Exclude source course itself
        sim_df = sim_df[sim_df['course_id'] != course_id]
        
        # Sort and merge course metadata
        top_candidates = sim_df.sort_values(by='similarity_score', ascending=False).head(top_n)
        result = top_candidates.merge(self.courses_df, on='course_id', how='left')
        return result

    def get_user_profile_vector(self, user_id: str) -> Optional[np.ndarray]:
        """Computes weighted average user profile vector from user interaction history."""
        user_weights = self.user_interaction_weights[self.user_interaction_weights['user_id'] == user_id]
        
        if user_weights.empty:
            return None
            
        weighted_vectors = []
        total_weight = 0.0
        
        for _, row in user_weights.iterrows():
            cid = row['course_id']
            w = row['weight']
            if cid in self.course_id_to_idx:
                c_idx = self.course_id_to_idx[cid]
                c_vec = self.tfidf_matrix[c_idx].toarray().flatten()
                weighted_vectors.append(w * c_vec)
                total_weight += w
                
        if not weighted_vectors or total_weight == 0:
            return None
            
        user_profile = np.sum(weighted_vectors, axis=0) / total_weight
        return user_profile

    def recommend(self, user_id: str, top_n: int = 5) -> pd.DataFrame:
        """Personalized recommendation: returns top_n courses matching user profile."""
        user_profile = self.get_user_profile_vector(user_id)
        
        # Cold-Start Fallback: If unknown/new user, return Popularity Recommender results
        if user_profile is None:
            fallback_recs = self.popularity_fallback.recommend(user_id, top_n=top_n)
            fallback_recs['recommendation_type'] = 'popularity_fallback'
            return fallback_recs
            
        # Compute similarity between user profile vector and all course vectors
        sim_scores = cosine_similarity(user_profile.reshape(1, -1), self.tfidf_matrix).flatten()
        
        rec_df = pd.DataFrame({
            'course_id': self.courses_df['course_id'],
            'similarity_score': sim_scores
        })
        
        # Candidate Filtering: Remove courses user has already interacted with
        interacted_courses = self.user_history.get(user_id, set())
        rec_df = rec_df[~rec_df['course_id'].isin(interacted_courses)]
        
        # Sort and merge course metadata
        top_candidates = rec_df.sort_values(by='similarity_score', ascending=False).head(top_n)
        result = top_candidates.merge(self.courses_df, on='course_id', how='left')
        result['recommendation_type'] = 'content_based'
        return result
