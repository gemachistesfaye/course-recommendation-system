# Content-Based Course Recommendation

## 1. Overview & Architecture
Content-Based Filtering recommends courses to learners by analyzing **course metadata** and matching it to a user's **past interaction history**.

### Pipeline Architecture:
```text
Course Metadata (title, category, skills, description)
      ↓
Feature Representation & Weighted Text Construction
      ↓
TF-IDF Vectorization (21 x 402 Matrix)
      ↓
User Interaction History (weighted implicit signals)
      ↓
User Profile Vector (weighted average course vector)
      ↓
Cosine Similarity Scoring
      ↓
Candidate Filtering (exclude already-interacted courses)
      ↓
Top-N Ranking
```

---

## 2. Course Feature Representation
We construct a combined text string per course from:
- `title` (weighted 2x)
- `category` (weighted 2x)
- `skills` (weighted 2x)
- `description` (weighted 1x)

---

## 3. Text Vectorization: TF-IDF
- **TF (Term Frequency):** Frequency of term $t$ in a course text document.
- **IDF (Inverse Document Frequency):** Log-scaled ratio of total courses over courses containing term $t$.
- **Matrix Dimensions:** 21 courses $\times$ 402 learned terms (unigrams + bigrams).

---

## 4. Distance Metric: Cosine Similarity
$$\text{Cosine Similarity}(\mathbf{u}, \mathbf{v}) = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\|_2 \|\mathbf{v}\|_2}$$
Measures the directional angle between the user profile vector $\mathbf{u}$ and course vector $\mathbf{v}$ (ranging from $0.0$ to $1.0$).

---

## 5. User Profile Vectors
User interest profiles are constructed via weighted averaging of interacted course vectors:
$$\mathbf{u} = \frac{\sum w_i \mathbf{v}_i}{\sum w_i}$$
where $w_{\text{VIEW}} = 1$, $w_{\text{BOOKMARK}} = 2$, $w_{\text{ENROLL}} = 3$, $w_{\text{COMPLETE}} = 4$.

---

## 6. Cold-Start Handling & Fallback
- **New Course Cold Start:** Fully supported! A newly added course can be recommended immediately based on its metadata text.
- **New User Cold Start:** For users with 0 interactions, the system falls back to the Phase 1 `PopularityRecommender`.

---

## 7. Limitations of Content-Based Filtering
1. **Overspecialization / Limited Serendipity:** Only recommends items similar to past history (e.g. a Python user never discovers Figma UI courses).
2. **Metadata Dependency:** Quality depends entirely on descriptive, high-quality course metadata.
3. **Ignores Community Behavior:** Does not leverage patterns from similar users (addressed by Collaborative Filtering in Phase 3).
