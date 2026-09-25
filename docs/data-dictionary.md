# Data Dictionary & Signal Weighting Specification

## 1. Dataset Schemas

### `users.csv`
| Column Name | Data Type | Meaning | Example | Required | Constraints |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `user_id` | String | Unique user identifier | `U001` | Yes | Primary key, pattern `U\d{3}` |
| `name` | String | Learner full name | `Learner 1` | Yes | Non-empty |
| `persona` | String | Target career background | `Frontend Developer` | Yes | High-level role grouping |
| `preferred_categories` | String (CSV) | Domain interest tags | `Web Development,Design` | Yes | Comma-separated categories |

### `courses.csv`
| Column Name | Data Type | Meaning | Example | Required | Constraints |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `course_id` | String | Unique course identifier | `C01` | Yes | Primary key, pattern `C\d{2}` |
| `title` | String | Official course title | `React 18 Masterclass` | Yes | Non-empty |
| `description` | String | Summary of curriculum | `Learn key concepts...` | Yes | Text representation |
| `category` | String | Primary topic category | `Web Development` | Yes | Categorical domain tag |
| `difficulty` | String | Target skill level | `Beginner` | Yes | `Beginner`, `Intermediate`, `Advanced` |
| `skills` | String (CSV) | Technical skill keywords | `React,Redux,JavaScript` | Yes | Comma-separated skill tags |

### `interactions.csv`
| Column Name | Data Type | Meaning | Example | Required | Constraints |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `user_id` | String | Foreign key to `users.csv` | `U001` | Yes | Must match valid `user_id` |
| `course_id` | String | Foreign key to `courses.csv` | `C01` | Yes | Must match valid `course_id` |
| `interaction_type` | String | Event type name | `VIEW` | Yes | `VIEW`, `BOOKMARK`, `ENROLL`, `COMPLETE`, `RATE` |
| `rating` | Float / Null | Explicit 1-5 star score | `4.5` | No | Populated only for `RATE` interactions |

---

## 2. Interaction Events & Signal Classification

| Interaction Type | Signal Category | Description |
| :--- | :--- | :--- |
| `VIEW` | Implicit | Learner visited the course page. |
| `BOOKMARK` | Implicit | Learner saved course to personal wishlist. |
| `ENROLL` | Implicit | Learner registered for the course. |
| `COMPLETE` | Implicit | Learner finished 100% of course content. |
| `RATE` | Explicit | Learner submitted explicit star rating (2.5 to 5.0). |

---

## 3. Experimental Interaction Signal Weights

For converting multi-event implicit interaction logs into single numerical values per user-course pair, we assign the following heuristic weights:

$$\text{VIEW} = 1.0, \quad \text{BOOKMARK} = 2.0, \quad \text{ENROLL} = 3.0, \quad \text{COMPLETE} = 4.0$$

> [!IMPORTANT]
> **Modeling Assumption Warning:**
> These interaction weights are **experimental assumptions** designed for initial numerical representation. They are not universal correct constants. In future phases, these weights can be hyperparameter-tuned, normalized, or evaluated against implicit interaction alternative models.
