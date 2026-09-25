# Content-Based Course Recommendation Engine

## 1. Algorithmic Overview
Content-Based Filtering recommends courses to a learner by matching the textual content metadata of available courses with the learner's historical preference profile.

## 2. Pipeline Architecture
Course Metadata -> Feature Engineering (weighted text) -> TF-IDF Vectorization (21 x 402 Matrix V) -> User Interaction Aggregation -> User Profile Vector (u = sum(w_i * v_i) / sum(w_i)) -> Cosine Similarity -> Candidate Filtering -> Ranking & Top-N Selection

## 3. Mathematical Formulation
- Feature Engineering: Combined Text = (Title * 2) + (Category * 2) + (Skills * 2) + (Description * 1)
- TF-IDF Vectorization: TF(t, d) * IDF(t, D)
- User Profile Vector: Weighted average of course vectors based on implicit weights (VIEW=1, BOOKMARK=2, ENROLL=3, COMPLETE=4).
- Cosine Similarity: cos(u, v) = (u . v) / (||u|| * ||v||)

## 4. Cold-Start Strategy
Fallback to Popularity Baseline for unknown/new users.
