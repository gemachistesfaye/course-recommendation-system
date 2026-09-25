# Data Dictionary & Signal Weighting Specification

## 1. Dataset Schemas
- users.csv (50 records): user_id, name, persona, preferred_categories
- courses.csv (21 records): course_id, title, description, category, difficulty, skills
- interactions.csv (682 records): user_id, course_id, interaction_type, rating

## 2. Interaction Weights
VIEW = 1.0, BOOKMARK = 2.0, ENROLL = 3.0, COMPLETE = 4.0
