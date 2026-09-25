# Recommendation System Fundamentals

## 1. Core Entities
Every recommendation system models the relationships between three core domain entities:

- **Users ($U$):** The active entities receiving recommendations (e.g., students/learners).
- **Items ($I$):** The catalog items available to be recommended (e.g., technology courses).
- **Interactions ($E$):** Observed actions taken by users on items, forming events $e = (u, i, t, r)$ where $u \in U$, $i \in I$, $t$ is interaction type, and $r$ is an optional rating.

```
       +----------+                 +----------+
       | Users (U)|                 | Items (I)|
       +----------+                 +----------+
            \                             /
             \                           /
              \                         /
               v                       v
            +-----------------------------+
            |      Interactions (E)       |
            | (VIEW, BOOKMARK, ENROLL,    |
            |   COMPLETE, RATE)           |
            +-----------------------------+
```

---

## 2. Explicit vs. Implicit Feedback

| Characteristic | Explicit Feedback | Implicit Feedback |
| :--- | :--- | :--- |
| **Definition** | Direct, intentional preference scores provided by the user. | Passive behavioral signals collected from user actions. |
| **Examples in System** | Star ratings (`RATE`: 2.5 to 5.0 stars). | `VIEW`, `BOOKMARK`, `ENROLL`, `COMPLETE`. |
| **Volume & Sparsity** | Extremely sparse; most users consume content without rating. | Abundant; captured automatically as users browse. |
| **Signal Noise** | Low noise; clear positive or negative sentiment. | High noise; viewing a course does not guarantee satisfaction. |
| **Negative Feedback** | Low ratings (e.g., 1 or 2 stars). | Missing actions (hard to distinguish disinterest from unwareness). |

---

## 3. Recommendation vs. Search vs. Classification

- **Search:** Driven by an **explicit query string** (e.g., *"Node.js microservices"*). The system computes keyword relevancy scores against the catalog.
- **Classification:** Predicts a discrete category label for an instance (e.g., classifying whether a course is *"Beginner"* or *"Advanced"*).
- **Recommendation:** Operates without an explicit query string. The system predicts what items a specific user is likely to prefer based on their interaction history and item characteristics.

---

## 4. The Candidate Generation & Ranking Funnel

Modern industrial recommendation architectures process items through a multi-stage funnel:

```
                  +-----------------------------------+
                  |      Full Catalog (All Courses)   |
                  +-----------------------------------+
                                    |
                                    v
                  +-----------------------------------+
                  | Stage 1: Candidate Generation     |
                  | Filter out previously consumed    |
                  | courses or ineligible items       |
                  +-----------------------------------+
                                    |
                                    v
                  +-----------------------------------+
                  | Stage 2: Scoring                  |
                  | Compute numerical match scores    |
                  | (Popularity, TF-IDF Cosine, SVD)  |
                  +-----------------------------------+
                                    |
                                    v
                  +-----------------------------------+
                  | Stage 3: Ranking & Selection      |
                  | Sort descending & return Top-N    |
                  +-----------------------------------+
```

---

## 5. The User-Item Interaction Matrix & Sparsity

The interaction history of $|U|$ users across $|I|$ items is represented as a 2D matrix $M \in \mathbb{R}^{|U| \times |I|}$:

$$M_{u,i} = \text{Aggregated Interaction Weight of User } u \text{ on Course } i$$

### Matrix Sparsity Formula:
$$\text{Sparsity} = \left(1 - \frac{\text{Number of Observed Interactions}}{|U| \times |I|}\right) \times 100\%$$

In real-world recommendation engines, sparsity routinely exceeds $99\%$. Recommender algorithms (such as Matrix Factorization) excel at predicting the missing values in this sparse matrix.

---

## 6. Key Recommender Challenges
- **User Cold-Start:** Brand-new users have 0 past interactions, making it impossible to compute personalized preference vectors.
- **Item Cold-Start:** Brand-new courses have 0 student interactions, preventing collaborative filtering models from learning their audience.
- **Popularity Baseline:** Recommends the overall most-interacted courses across all users. It serves two vital functions:
  1. Acts as a baseline benchmark that all advanced ML models must outperform.
  2. Provides a reliable fallback strategy for new users during cold-start.
