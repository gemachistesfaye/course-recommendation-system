# Project Overview: Personalized Course Recommendation System

## 1. Executive Summary
The **Personalized Course Recommendation System** is an end-to-end Machine Learning project designed to solve choice overload in online education platforms. By analyzing learner interactions and course metadata, the system dynamically recommends personalized learning paths to students.

The project is developed iteratively across 5 distinct phases, transitioning from foundational data engineering to advanced ML modeling, offline evaluation, and production service deployment.

```
                  +-----------------------------------+
                  |   Learner Interactions & Catalog  |
                  +-----------------------------------+
                                    |
                                    v
                  +-----------------------------------+
                  |    Phase 1: Foundation & Baseline |
                  |    (User-Item Matrix & Popularity)|
                  +-----------------------------------+
                                    |
                                    v
                  +-----------------------------------+
                  |    Phase 2: Content-Based Engine  |
                  |    (TF-IDF & User Profile Vector) |
                  +-----------------------------------+
                                    |
                                    v
                  +-----------------------------------+
                  |    Phase 3: Collaborative Engine  |
                  |    (Matrix Factorization / SVD)   |
                  +-----------------------------------+
                                    |
                                    v
                  +-----------------------------------+
                  |    Phase 4: Hybrid & Offline Eval |
                  |    (Precision@K, Recall@K, NDCG)  |
                  +-----------------------------------+
                                    |
                                    v
                  +-----------------------------------+
                  |    Phase 5: Serving & FastAPI     |
                  +-----------------------------------+
```

---

## 2. Problem Statement
Learners on modern online learning platforms face thousands of tech courses spanning web development, data science, machine learning, cloud engineering, and cybersecurity. 

Without personalized guidance:
- Learners struggle with **choice overload** and drop out.
- Traditional **keyword search** requires users to know exact technical terms in advance.
- High-quality niche courses remain undiscovered under standard popularity rankings.

A personalized recommendation engine dynamically presents relevant courses tailored to each student's unique skill level, background, and historical learning behavior.

---

## 3. Project Goals & Learning Objectives
1. **Algorithmic Mastery:** Build, evaluate, and compare fundamental recommenders (Popularity Baseline, Content-Based Filtering, Collaborative Filtering, and Hybrid models).
2. **Production-Grade Software Design:** Organize reusable ML modules under a clean Python package (`ml/src/`) supported by automated pytest suites.
3. **Rigorous Offline Evaluation:** Measure recommendation quality using industry-standard metrics: Precision@K, Recall@K, and NDCG@K under Leave-Last-Course-Out validation.
4. **End-to-End System Architecture:** Prepare the ML engine for eventual REST API serving via FastAPI, Node.js/Express, PostgreSQL, and React.

---

## 4. Scope & System Boundaries

### In-Scope (ML Architecture):
- Data generation, cleaning, and matrix pivot transformations.
- Popularity Baseline Recommender.
- Content-Based Recommendation using TF-IDF feature vectors and cosine similarity.
- Collaborative Filtering (User-Item Matrix Factorization & KNN).
- Hybrid Recommendation Ensembles.
- Offline Evaluation pipelines with data-leakage prevention.

### Out-of-Scope (Phase 1–2 Non-Goals):
- Production web deployment, React UI, Node.js Express backend, PostgreSQL database instances, Docker containerization, or live streaming APIs.

---

## 5. Development Roadmap & Status

| Phase | Description | Key Modules / Outputs | Status |
| :--- | :--- | :--- | :--- |
| **Phase 1** | Recommendation Fundamentals & Data Foundation | `users.csv`, `courses.csv`, `interactions.csv`, User-Item Matrix, `PopularityRecommender` | **VERIFIED** |
| **Phase 2** | Content-Based Filtering & Text Vectorization | `feature_builder.py`, `TfidfVectorizer`, Cosine Similarity, `ContentBasedRecommender`, Offline Evaluator | **VERIFIED** |
| **Phase 3** | Collaborative Filtering & Matrix Factorization | User/Item KNN, Matrix Factorization (SVD) | Planned |
| **Phase 4** | Hybrid Recommendation & Comprehensive Offline Evaluation | Weighted Ensembles, Precision@K, Recall@K, NDCG@K | Planned |
| **Phase 5** | Serving, API Integration, & System Architecture | FastAPI inference endpoint, Docker packaging | Planned |

---

## 6. Technology Stack
- **Language:** Python 3.11
- **Data Engineering:** Pandas, NumPy
- **Machine Learning & NLP:** Scikit-Learn (TF-IDF Vectorizer, Cosine Similarity)
- **Testing:** Pytest
- **Future Serving Stack:** React, TypeScript, Node.js, Express, PostgreSQL, Prisma, FastAPI
