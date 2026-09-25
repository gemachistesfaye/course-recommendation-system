# Recommendation System Fundamentals

## 1. Core Entities: Users, Items, and Interactions
- **Users ($U$):** The active learners receiving recommendations.
- **Items ($I$):** The tech courses available in the catalog (e.g., Python, React, Cybersecurity).
- **Interactions ($E$):** Observed actions taken by users on courses (e.g., viewing, bookmarking, enrolling, completing, rating).

## 2. Explicit vs. Implicit Feedback
- **Explicit Feedback:** Direct ratings provided intentionally by users (e.g., 4.5 star rating).
  - *Pros:* High quality sentiment signal.
  - *Cons:* Extremely sparse as most users do not rate courses.
- **Implicit Feedback:** Behavioral signals automatically captured by user activity logs (e.g., views, bookmarks, enrollments, completions).
  - *Pros:* Abundant, passive collection.
  - *Cons:* Noisy (viewing a course page does not guarantee satisfaction).

## 3. Recommendation vs. Search vs. Classification
| Aspect | Search | Classification | Recommendation |
| :--- | :--- | :--- | :--- |
| **Input** | Explicit query text | Structured feature vector | User identity & past behavior |
| **Goal** | Match query intent | Assign discrete class label | Predict unstated preference |
| **Output** | Query-relevant items | Single category label | Ranked list of Top-N items |

## 4. The Candidate Generation & Ranking Pipeline
1. **Candidate Generation (Retrieval):** Filter out previously consumed or non-eligible courses.
2. **Scoring:** Compute predicted preference scores for candidates using statistical or ML models.
3. **Ranking:** Sort scored candidates in descending order.
4. **Top-N Selection:** Select the top $N$ items to present on the user interface.

## 5. Core Challenges
- **Sparsity:** Users interact with only a tiny percentage of the total catalog, leaving most cells in the User-Item matrix empty.
- **Cold-Start Problem:**
  - *New User:* Zero interaction history available to compute personalized preferences.
  - *New Item:* Newly added course has zero interactions to calculate collaborative signals.
- **Popularity Baseline:** Recommends globally popular courses. Serves as a strong benchmark for ML models and solves the new-user cold-start challenge.
