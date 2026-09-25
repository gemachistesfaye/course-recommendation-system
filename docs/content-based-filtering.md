# Content-Based Course Recommendation Engine

## 1. Algorithmic Overview
Content-Based Filtering recommends courses to a learner by matching the **textual content metadata** of available courses with the learner's **historical preference profile**.

Unlike Collaborative Filtering (which relies on shared user behavior), Content-Based Filtering models item properties directly.

---

## 2. Complete Phase 2 Architecture & Pipeline

```
 Course Metadata (title, category, skills, description)
       ↓
 1. Text Feature Engineering & Repetition Weighting
       ↓
 2. TF-IDF Vectorization (21 x 402 Matrix V)
       ↓
 3. User Interaction Aggregation (implicit weights w_i)
       ↓
 4. User Profile Vector Construction (u = sum(w_i * v_i) / sum(w_i))
       ↓
 5. Cosine Similarity Scoring (sim(u, v_i))
       ↓
 6. Candidate Filtering (Exclude previously interacted courses)
       ↓
 7. Ranking & Top-N Selection (Popularity Fallback for Cold-Start)
```

---

## 3. Step-by-Step Mathematical Formulation

### Step A: Feature Engineering & Repetition Weighting
To give critical domain metadata higher relative influence before vector L2-normalization, course text fields are cleaned and repeated according to configurable feature weights:

$$\text{Combined Text} = (\text{Title} \times 2) + (\text{Category} \times 2) + (\text{Skills} \times 2) + (\text{Description} \times 1)$$

### Step B: TF-IDF Vectorization
Text strings are converted into TF-IDF vectors where each dimension represents a specific unigram or bigram term $t$:

$$\text{TF-IDF}(t, d, D) = \text{TF}(t, d) \times \text{IDF}(t, D)$$

$$\text{TF}(t, d) = \frac{\text{Count of term } t \text{ in course } d}{\text{Total terms in course } d}$$

$$\text{IDF}(t, D) = \ln\left(\frac{|D|}{|\{d \in D : t \in d\}|}\right) + 1$$

- **Corpus Matrix $\mathbf{V}$:** Shape $(21, 402)$ representing 21 courses across 402 learned vocabulary features.

### Step C: User Profile Vector Construction
For a user $u$ with interaction history $H_u = \{(c_1, w_1), (c_2, w_2), \dots, (c_k, w_k)\}$, their preference profile vector $\mathbf{u} \in \mathbb{R}^{402}$ is calculated as a weighted average of course vectors:

$$\mathbf{u} = \frac{\sum_{i=1}^{k} w_i \mathbf{v}_{c_i}}{\sum_{i=1}^{k} w_i}$$

where implicit weights $w_i \in \{1.0, 2.0, 3.0, 4.0\}$ correspond to `VIEW`, `BOOKMARK`, `ENROLL`, and `COMPLETE`.

### Step D: Cosine Similarity Scoring
The match score between user profile vector $\mathbf{u}$ and candidate course vector $\mathbf{v}_c$ is calculated via Cosine Similarity:

$$\text{Score}(u, c) = \cos(\mathbf{u}, \mathbf{v}_c) = \frac{\mathbf{u} \cdot \mathbf{v}_c}{\|\mathbf{u}\|_2 \|\mathbf{v}_c\|_2}$$

---

## 4. Candidate Filtering & Cold-Start Strategy

1. **Item-to-Item Recommendation (`recommend_similar_courses`):**
   Given a source course $c_{\text{source}}$, computes cosine similarity against all other courses and **strictly excludes $c_{\text{source}}$** from the result set.

2. **Personalized User Recommendation (`recommend`):**
   Filters out all course IDs previously interacted with by user $u$ ($c \notin H_u$).

3. **Cold-Start Fallback:**
   For a new user with 0 interaction history ($\mathbf{u} = \text{None}$), the engine seamlessly falls back to the `PopularityRecommender` baseline, labeling output items as `popularity_fallback`.

---

## 5. Strengths & Known Limitations

### Strengths:
- **No Item Cold-Start:** Newly created courses with 0 student enrollments can be recommended immediately based on text metadata.
- **High Explainability:** Easy to justify recommendations (*"Recommended because you completed Python Programming"*).
- **Domain Independence:** Does not require interaction logs from thousands of active users.

### Limitations:
- **Overspecialization (No Serendipity):** Only recommends items similar to past history (e.g., a Python developer is never recommended a Figma design course).
- **Metadata Dependent:** Quality is bounded by the richness of course text fields.
- **Ignores Community Preferences:** Does not leverage collaborative signals from similar learners.
