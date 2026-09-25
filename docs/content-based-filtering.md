# Content-Based Course Recommendation

## 1. Overview & Architecture
Content-Based Filtering recommends courses to learners by analyzing course metadata and matching it to a user\'s past interaction history.

### Pipeline Architecture:
Course Metadata -> Feature Representation -> TF-IDF Vectorization (21 x 402 Matrix) -> User Interaction History -> User Profile Vector -> Cosine Similarity -> Candidate Filtering -> Top-N Ranking

## 2. Course Feature Representation
Combined text string constructed from title (2x weight), category (2x weight), skills (2x weight), and description (1x weight).

## 3. TF-IDF & Distance Metric
- TF-IDF Matrix Shape: (21, 402)
- Distance Metric: Cosine Similarity between User Profile Vector and Course Vectors.

## 4. User Profile Vectors
Weighted average of interacted course vectors: u = sum(w_i * v_i) / sum(w_i).

## 5. Cold Start & Fallback
Fallback to Popularity Baseline for unknown/new users.
