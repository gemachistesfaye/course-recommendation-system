import pandas as pd
import re
from typing import List, Dict

REQUIRED_COURSE_COLUMNS = {'course_id', 'title', 'category', 'description', 'skills'}

def validate_course_features(df_courses: pd.DataFrame) -> None:
    """Validates that courses DataFrame contains all required metadata columns."""
    missing = REQUIRED_COURSE_COLUMNS - set(df_courses.columns)
    if missing:
        raise ValueError(f"df_courses missing required metadata columns: {missing}")

def clean_text(text: str) -> str:
    """Cleans text string by lowercasing, converting commas/hyphens to spaces, and stripping extra whitespace."""
    if not isinstance(text, str) or pd.isna(text):
        return ""
    cleaned = re.sub(r'[,/-]', ' ', text.lower())
    cleaned = re.sub(r'[^a-z0-9\s]', '', cleaned)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned

def build_combined_course_features(
    df_courses: pd.DataFrame,
    title_weight: int = 2,
    category_weight: int = 2,
    skills_weight: int = 2,
    description_weight: int = 1
) -> pd.Series:
    """
    Combines course metadata fields into a single text representation per course.
    Feature weights determine how many times each field is repeated in the text representation.
    """
    validate_course_features(df_courses)
    
    combined_texts = []
    
    for _, row in df_courses.iterrows():
        title = clean_text(row['title'])
        category = clean_text(row['category'])
        skills = clean_text(row['skills'])
        description = clean_text(row['description'])
        
        parts = []
        if title and title_weight > 0:
            parts.extend([title] * title_weight)
        if category and category_weight > 0:
            parts.extend([category] * category_weight)
        if skills and skills_weight > 0:
            parts.extend([skills] * skills_weight)
        if description and description_weight > 0:
            parts.extend([description] * description_weight)
            
        combined_str = " ".join(parts)
        combined_texts.append(combined_str)
        
    return pd.Series(combined_texts, index=df_courses['course_id'], name='combined_features')
