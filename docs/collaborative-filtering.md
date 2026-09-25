# Collaborative Filtering Course Recommendation Engine

## 1. Algorithmic Overview
Collaborative Filtering (CF) predicts a learner's course preferences by leveraging **community interaction patterns** across users and items, rather than relying on textual item metadata.

```
       +-------------------------------------------------------+
       | Raw Interaction Event Logs (interactions.csv)         |
       +-------------------------------------------------------+
                                   |
                                   v
       +-------------------------------------------------------+
       | 2D User-Item Interaction Matrix R (50 Users x 21 Courses)|
       | Density: 27.24% | Sparsity: 72.76%                    |
       +-------------------------------------------------------+
               /                   |                   \
              /                    |                    \
             v                     v                     v
   +-------------------+ +-------------------+ +-----------------------+
   | User-User CF      | | Item-Item CF      | | Matrix Factorization  |
   | (Cosine User Sim) | | (Cosine Item Sim) | | (Truncated SVD)     |
   +-------------------+ +-------------------+ +-----------------------+
```

---

## 2. Mathematical Formulation & Models

### A. User-Item Matrix Construction ($R$)
Implicit weights ($w_{\text{VIEW}}=1, w_{\text{BOOKMARK}}=2, w_{\text{ENROLL}}=3, w_{\text{COMPLETE}}=4$) are aggregated per $(u, i)$ pair:
$$R_{u, i} = \sum_{e \in \text{Events}(u, i)} w_e$$
Unobserved cells are set to $0.0$.

### B. User-User Collaborative Filtering
For target user $u$, finds top $K$ most similar users $v$ using Cosine Similarity on user row vectors $\mathbf{r}_u$:
$$\text{Sim}(u, v) = \frac{\mathbf{r}_u \cdot \mathbf{r}_v}{\|\mathbf{r}_u\|_2 \|\mathbf{r}_v\|_2}$$
Candidate courses $c \notin H_u$ are scored as:
$$\text{Score}(u, c) = \frac{\sum_{v \in \text{Neighbors}} \text{Sim}(u, v) \times R_{v, c}}{\sum_{v \in \text{Neighbors}} \text{Sim}(u, v)}$$

### C. Item-Item Collaborative Filtering
Computes similarity between course column vectors $\mathbf{c}_i \in \mathbb{R}^{50}$:
$$\text{Sim}(c_1, c_2) = \frac{\mathbf{c}_1 \cdot \mathbf{c}_2}{\|\mathbf{c}_1\|_2 \|\mathbf{c}_2\|_2}$$

### D. Matrix Factorization via Truncated SVD
Decomposes user-item matrix $R$ into lower-rank latent factor matrices $P \in \mathbb{R}^{50 \times k}$ and $Q \in \mathbb{R}^{21 \times k}$ ($k=5$ factors):
$$R \approx P \times Q^T$$
Predicted preference matrix $\hat{R} = P Q^T$ reconstructs unobserved interaction cells without high-dimensional computational overhead.

---

## 3. Cold-Start & Fallback Strategy
- **New User Cold Start:** When an unknown user $u$ has 0 interaction rows, Collaborative Filtering models fall back to the `PopularityRecommender` baseline, tagging returned candidates as `popularity_fallback`.
- **New Item Cold Start:** Pure Collaborative Filtering cannot recommend a new course with 0 interactions (resolved by Content-Based Filtering or Hybrid Ensembles in Phase 4).
