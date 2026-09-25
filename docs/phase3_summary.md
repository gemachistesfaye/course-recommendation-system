# Phase 3 Summary: Collaborative Filtering & Matrix Factorization

## 1. Key Accomplishments
- Implemented `build_user_item_matrix()` generating a $50 \times 21$ 2D matrix ($27.24\%$ density, $72.76\%$ sparsity).
- Implemented `UserBasedRecommender` using K-Nearest Neighbor user similarity scoring.
- Implemented `ItemBasedRecommender` computing collaborative course co-consumption similarities.
- Implemented `MatrixFactorizationRecommender` using Truncated SVD low-rank latent decomposition ($k=5$).
- Built Leave-Last-Course-Out offline evaluation pipeline evaluating Precision@5 and Recall@5 across all 5 models.
- Created `test_collaborative.py` and passed full 14/14 unit test suite.

---

## 2. Deliverables Created
1. [`ml/src/collaborative/matrix_builder.py`](file:///c:/Users/HP/Desktop/ML-AI/recommendation_system/ml/src/collaborative/matrix_builder.py)
2. [`ml/src/collaborative/user_based.py`](file:///c:/Users/HP/Desktop/ML-AI/recommendation_system/ml/src/collaborative/user_based.py)
3. [`ml/src/collaborative/item_based.py`](file:///c:/Users/HP/Desktop/ML-AI/recommendation_system/ml/src/collaborative/item_based.py)
4. [`ml/src/collaborative/matrix_factorization.py`](file:///c:/Users/HP/Desktop/ML-AI/recommendation_system/ml/src/collaborative/matrix_factorization.py)
5. [`ml/src/collaborative/__init__.py`](file:///c:/Users/HP/Desktop/ML-AI/recommendation_system/ml/src/collaborative/__init__.py)
6. [`ml/notebooks/03_collaborative_filtering.ipynb`](file:///c:/Users/HP/Desktop/ML-AI/recommendation_system/ml/notebooks/03_collaborative_filtering.ipynb)
7. [`ml/tests/test_collaborative.py`](file:///c:/Users/HP/Desktop/ML-AI/recommendation_system/ml/tests/test_collaborative.py)
8. [`docs/collaborative-filtering.md`](file:///c:/Users/HP/Desktop/ML-AI/recommendation_system/docs/collaborative-filtering.md)
9. [`docs/phase3_summary.md`](file:///c:/Users/HP/Desktop/ML-AI/recommendation_system/docs/phase3_summary.md)

---

## 3. Comparative Offline Evaluation Results (Leave-Last-Course-Out, N=50 Users)

| Model | Precision@5 | Recall@5 | Total Hits | Recommendation Characteristic |
| :--- | :--- | :--- | :--- | :--- |
| **Content-Based (Phase 2)** | **0.1760** | **0.8800** | 44 | High metadata alignment |
| **Item-Item Collaborative** | **0.1560** | **0.7800** | 39 | Co-consumption similarity |
| **SVD Matrix Factorization** | **0.1360** | **0.6800** | 34 | Low-rank latent pattern extraction |
| **User-User Collaborative** | **0.1240** | **0.6200** | 31 | Peer neighbor aggregation |
| **Popularity Baseline** | **0.0560** | **0.2800** | 14 | Non-personalized top courses |

> [!IMPORTANT]
> **Synthetic Data Disclaimer:**
> These evaluation results are derived from our synthetic dataset (`interactions.csv`). They demonstrate that personalizing recommendations via content metadata or collaborative behavior significantly outperforms the non-personalized Popularity Baseline. They should **not** be interpreted as expected real-world production performance.

---

## 4. Test Suite Execution
Ran full test suite across Phase 1, Phase 2, and Phase 3 (`pytest ml/tests/`):
- **Result:** **14 passed in 4.63s (100% pass rate)**.
