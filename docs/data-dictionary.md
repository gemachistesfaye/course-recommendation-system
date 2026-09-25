# Data Dictionary & Signal Weighting Specification

## 1. Dataset Overview
The system processes three raw CSV datasets located in `ml/data/raw/`:
- `users.csv`: Learner profiles and career personas.
- `courses.csv`: Technology course metadata and technical skill tags.
- `interactions.csv`: Granular interaction event logs capturing learner funnel activity.

---

## 2. File Schemas & Entity Definitions

### `users.csv` (50 Records)
| Field Name | Data Type | Description | Example | Required | Constraints |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `user_id` | String | Unique learner identifier | `U001` | Yes | Primary Key, format `U\d{3}` |
| `name` | String | Learner full name | `Learner 1` | Yes | Non-empty string |
| `persona` | String | Target career background | `Frontend Developer` | Yes | 1 of 5 predefined tech personas |
| `preferred_categories` | String (CSV) | Domain interest tags | `Web Development,Design` | Yes | Comma-separated categories |

### `courses.csv` (21 Records)
| Field Name | Data Type | Description | Example | Required | Constraints |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `course_id` | String | Unique course identifier | `C01` | Yes | Primary Key, format `C\d{2}` |
| `title` | String | Official course title | `React 18 Masterclass` | Yes | Non-empty string |
| `description` | String | Curriculum text summary | `Learn key concepts...` | Yes | Text metadata |
| `category` | String | Primary topic domain | `Web Development` | Yes | 1 of 10 tech categories |
| `difficulty` | String | Target skill level | `Intermediate` | Yes | `Beginner`, `Intermediate`, `Advanced` |
| `skills` | String (CSV) | Technical keyword tags | `React,Redux,JavaScript` | Yes | Comma-separated skill keywords |

### `interactions.csv` (682 Records)
| Field Name | Data Type | Description | Example | Required | Constraints |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `user_id` | String | Foreign key to `users.csv` | `U001` | Yes | Must reference valid `user_id` |
| `course_id` | String | Foreign key to `courses.csv` | `C01` | Yes | Must reference valid `course_id` |
| `interaction_type` | String | Funnel event type name | `ENROLL` | Yes | `VIEW`, `BOOKMARK`, `ENROLL`, `COMPLETE`, `RATE` |
| `rating` | Float / Null | Explicit star score | `4.5` | No | Populated only for `RATE` events ($2.5 \le r \le 5.0$) |

---

## 3. Interaction Event Funnel & Signal Classification

Learner activity follows a natural conversion funnel:

$$\text{VIEW} \longrightarrow \text{BOOKMARK} \longrightarrow \text{ENROLL} \longrightarrow \text{COMPLETE} \longrightarrow \text{RATE}$$

| Event Type | Signal Category | Meaning | Weight ($w_i$) |
| :--- | :--- | :--- | :--- |
| `VIEW` | Implicit | Learner visited the course details page. | $1.0$ |
| `BOOKMARK` | Implicit | Learner saved the course to their wishlist. | $2.0$ |
| `ENROLL` | Implicit | Learner registered for the course. | $3.0$ |
| `COMPLETE` | Implicit | Learner completed 100% of course lectures. | $4.0$ |
| `RATE` | Explicit | Learner submitted a 1-5 star review score. | Reserved for Eval |

---

## 4. Modeling Assumptions & Disclaimers

1. **Experimental Weight Assignment:**
   The implicit weights ($w_{\text{VIEW}}=1, w_{\text{BOOKMARK}}=2, w_{\text{ENROLL}}=3, w_{\text{COMPLETE}}=4$) represent **heuristic modeling assumptions**. They provide an initial mechanism to convert multi-event behavioral logs into consolidated numerical signals. In future phases, these weights can be hyperparameter-tuned or replaced by implicit interaction models (e.g., ALS confidence scaling).

2. **Explicit Rating Handling:**
   Explicit star ratings (`RATE`) are intentionally excluded from implicit profile vector construction to avoid double-counting completed courses. Explicit ratings are reserved for offline evaluation in Phase 4.
