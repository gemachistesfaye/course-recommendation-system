# Personalized Course Recommendation System

An end-to-end Machine Learning learning project for personalized technology course recommendations.

## Project Structure
```text
course-recommendation-system/
├── ml/
│   ├── data/
│   │   ├── raw/                  # Contains users.csv, courses.csv, interactions.csv
│   │   ├── processed/            # Processed numerical summaries
│   │   └── generate_data.py      # Synthetic dataset generator
│   ├── notebooks/
│   │   └── 01_eda_and_interaction_matrix.ipynb # EDA & User-Item Matrix notebook
│   ├── src/
│   │   └── preprocessing/        # Reusable production modules
│   │       ├── data_loader.py    # Loaders, schema validators, weights, matrix creators
│   │       ├── recommender.py    # Popularity Baseline Recommender
│   │       └── __init__.py
│   └── tests/
│       └── test_preprocessing.py # Pytest unit tests
├── docs/
│   ├── data-dictionary.md
│   ├── phase1_summary.md
│   ├── project-overview.md
│   └── recommendation-fundamentals.md
└── README.md
```

## Running Unit Tests
```bash
pytest ml/tests/test_preprocessing.py
```

## Dataset Statistics (Phase 1 Data Foundation)
- **Users:** 50 unique learners across 5 tech personas
- **Courses:** 21 courses covering Web Dev, Python, ML, AI, Databases, Cloud, Cybersecurity, Mobile, and UI/UX
- **Interactions:** 682 funnel events (`VIEW`, `BOOKMARK`, `ENROLL`, `COMPLETE`, `RATE`)
- **Matrix Sparsity:** 72.76%
