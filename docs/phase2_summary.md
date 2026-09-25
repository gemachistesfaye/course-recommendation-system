# Phase 2 Summary: Content-Based Course Recommendation

## 1. Key Accomplishments
- **Feature Engineering:** Built combined text feature strings with title, category, skills (2x weights), and description (1x weight).
- **TF-IDF Vectorization:** Fitted `TfidfVectorizer` (unigrams + bigrams) generating a $(21, 402)$ numerical corpus matrix.
- **Item-to-Item Similarity:** Implemented `recommend_similar_courses(course_id)` using cosine similarity, strictly excluding the source course from recommendations.
- **User Profile Vector Construction:** Implemented weighted user preference profile aggregation $\mathbf{u} = \frac{\sum w_i \mathbf{v}_i}{\sum w_i}$.
- **Personalized Recommender Class:** Built `ContentBasedRecommender` in `ml/src/content_based/recommender.py` with candidate filtering and cold-start fallback to `PopularityRecommender`.
- **Offline Evaluation:** Evaluated Precision@5 & Recall@5 under Leave-Last-Course-Out cross-validation across 50 users.

---

## 2. Deliverables Created
1. [`ml/src/content_based/feature_builder.py`](file:///c:/Users/HP/Desktop/ML-AI/recommendation_system/ml/src/content_based/feature_builder.py)
2. [`ml/src/content_based/recommender.py`](file:///c:/Users/HP/Desktop/ML-AI/recommendation_system/ml/src/content_based/recommender.py)
3. [`ml/src/content_based/__init__.py`](file:///c:/Users/HP/Desktop/ML-AI/recommendation_system/ml/src/content_based/__init__.py)
4. [`ml/notebooks/02_content_based_filtering.ipynb`](file:///c:/Users/HP/Desktop/ML-AI/recommendation_system/ml/notebooks/02_content_based_filtering.ipynb)
5. [`ml/tests/test_content_based.py`](file:///c:/Users/HP/Desktop/ML-AI/recommendation_system/ml/tests/test_content_based.py)
6. [`docs/content-based-filtering.md`](file:///c:/Users/HP/Desktop/ML-AI/recommendation_system/docs/content-based-filtering.md)
7. [`docs/phase2_summary.md`](file:///c:/Users/HP/Desktop/ML-AI/recommendation_system/docs/phase2_summary.md)

---

## 3. Example Execution Results

### Item-to-Item Similar Courses for `C06` (*Python Programming Masterclass*)
*(Note: Source course `C06` is excluded)*
1. `C07` - *Data Analysis with Pandas and NumPy* (Similarity: $0.1342$)
2. `C09` - *Deep Learning & PyTorch Specialization* (Similarity: $0.0568$)
3. `C10` - *Generative AI & LLM Application Building* (Similarity: $0.0558$)
4. `C08` - *Machine Learning A-Z: Hands-on Scikit-Learn* (Similarity: $0.0545$)
5. `C02` - *React 18 & Redux Toolkit Masterclass* (Similarity: $0.0278$)

### Offline Evaluation Results (Leave-Last-Course-Out, N=50 Users)
- **Content-Based Recommender:** Precision@5 = **0.1760** | Recall@5 = **0.8800**
- **Popularity Baseline:** Precision@5 = **0.0560** | Recall@5 = **0.2800**

> [!IMPORTANT]
> **Synthetic Data Disclaimer:**
> The metrics above (Precision@5 = 0.1760, Recall@5 = 0.8800) are derived from our synthetic persona-driven dataset (`interactions.csv`). They demonstrate that the model effectively captures the artificial interest clusters created during data generation. These metrics should **not** be interpreted as expected real-world production performance.

---

## 4. Test Verification
Ran full test suite across Phase 1 and Phase 2 (`pytest ml/tests/test_preprocessing.py ml/tests/test_content_based.py`).
- **Result:** **9 passed in 6.63s (100% pass rate)**.

---

## 5. Lessons Learned & Next Steps Motivation
While Content-Based Filtering achieves strong personalization and solves new-item cold start, it cannot recommend courses outside a user's explicit past metadata domain (overspecialization). This motivates **Phase 3: Collaborative Filtering**, where we leverage community behavior to discover unexpected, highly relevant recommendations!
