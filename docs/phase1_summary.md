# Phase 1 Summary: Fundamentals & Data Foundation

## Key Concepts Mastered
- **Recommendation Problem:** Predicting user preference without explicit query inputs.
- **Explicit vs. Implicit Signals:** Ratings vs. Funnel behavior (`VIEW`=1, `BOOKMARK`=2, `ENROLL`=3, `COMPLETE`=4).
- **Candidate Funnel:** Candidates -> Filter -> Score -> Rank -> Top-N.
- **Sparsity & Cold-Start:** Handling missing user-item pairs and new users via popular defaults.

## Deliverables Created
1. `ml/data/raw/` (courses.csv, users.csv, interactions.csv)
2. `ml/notebooks/01_eda_and_interaction_matrix.ipynb`
3. `ml/src/preprocessing/data_loader.py` & `recommender.py`
4. `ml/tests/test_preprocessing.py` (Passed 5/5 tests)
