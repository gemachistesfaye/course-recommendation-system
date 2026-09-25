# Recommendation System Fundamentals

## 1. Core Entities: Users, Items, and Interactions
- Users (U): The active learners receiving recommendations.
- Items (I): The tech courses available in the catalog (e.g., Python, React, Cybersecurity).
- Interactions (E): Observed actions taken by users on courses (e.g., viewing, bookmarking, enrolling, completing, rating).

## 2. Explicit vs. Implicit Feedback
- Explicit Feedback: Direct ratings provided intentionally by users (e.g., 4.5 star rating).
- Implicit Feedback: Behavioral signals automatically captured by user activity logs (e.g., views, bookmarks, enrollments, completions).

## 3. Recommendation vs. Search vs. Classification
- Search: Explicit query matching.
- Classification: Assigning discrete class labels.
- Recommendation: Predicting unstated preference scores and ranking Top-N items.

## 4. The Candidate Generation & Ranking Pipeline
1. Candidate Generation (Retrieval): Filter out previously consumed or non-eligible courses.
2. Scoring: Compute predicted preference scores for candidates using statistical or ML models.
3. Ranking: Sort scored candidates in descending order.
4. Top-N Selection: Select the top N items to present on the user interface.

## 5. Core Challenges
- Sparsity: Users interact with only a tiny percentage of the total catalog.
- Cold-Start Problem: New users or new items with zero interaction history.
- Popularity Baseline: Recommends globally popular courses as a benchmark and cold-start fallback.
