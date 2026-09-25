# Project Overview: Personalized Course Recommendation System

## 1. Problem Statement
Learners on online education platforms face choice overload when searching for courses that align with their skill level, interests, and past learning behavior. Traditional keyword search requires active effort and clear query formulation. A personalized recommendation system dynamically presents relevant tech courses to learners, driving engagement, course completion rates, and personalized learning paths.

## 2. Project Objectives
- Build an end-to-end recommendation engine starting from baseline data processing to advanced Machine Learning algorithms.
- Evaluate recommenders using industry-standard metrics: Precision@K, Recall@K, and NDCG@K.
- Ensure modular, production-ready python package structure (`ml/src/`).
- Design for future integration with a React + Node.js + PostgreSQL application stack.

## 3. Scope & Non-Goals
### In Scope for ML Phases:
- Data generation, cleaning, and user-course interaction matrix construction.
- Baseline Recommender (Popularity-based).
- Content-Based Filtering (TF-IDF, Cosine Similarity).
- Collaborative Filtering (Matrix Factorization, User/Item-based KNN).
- Hybrid Recommendation Engine.
- Offline Evaluation pipeline.

### Out of Scope for Phase 1:
- Frontend UI (React/TypeScript), Node.js Express backend, FastAPI ML serving layer, Docker containerization, or production deployment.

## 4. Planned Development Phases
- **Phase 1:** Recommendation System Fundamentals & Data Foundation *(COMPLETED & AUDITED)*
- **Phase 2:** Content-Based Filtering & Text Vectorization
- **Phase 3:** Collaborative Filtering & Matrix Factorization
- **Phase 4:** Hybrid Recommendation & Offline Evaluation (Precision@K, Recall@K, NDCG@K)
- **Phase 5:** Serving, API Integration, & System Architecture

## 5. Technology Stack
- **Language:** Python 3.11
- **Data Analysis & Processing:** Pandas, NumPy
- **Machine Learning & NLP:** Scikit-Learn
- **Testing:** Pytest
- **Future Integration:** React, Node.js, Express, PostgreSQL, Prisma, FastAPI
