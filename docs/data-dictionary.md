# Data Dictionary & Signal Weighting Specification

## 1. Dataset Schemas

### users.csv
- user_id: Unique user identifier (e.g. U001)
- name: Learner full name
- persona: Target career background
- preferred_categories: Domain interest tags

### courses.csv
- course_id: Unique course identifier (e.g. C01)
- title: Official course title
- description: Summary of curriculum
- category: Primary topic category
- difficulty: Skill level (Beginner, Intermediate, Advanced)
- skills: Technical skill keywords

### interactions.csv
- user_id: Foreign key to users.csv
- course_id: Foreign key to courses.csv
- interaction_type: Event type (VIEW, BOOKMARK, ENROLL, COMPLETE, RATE)
- rating: Explicit 1-5 star score (populated for RATE events)

## 2. Experimental Interaction Signal Weights
VIEW = 1.0, BOOKMARK = 2.0, ENROLL = 3.0, COMPLETE = 4.0
Note: These weights are experimental modeling assumptions to be evaluated in future phases.
