from pydantic import BaseModel, Field
from typing import Optional, List

class UserStory(BaseModel):
    feature_request: str
    generated_story: Optional[str] = None
    reviewed_story: Optional[str] = None
    human_approved: int = 0
    jira_ticket: Optional[str] = None
    review_cycles: int = 0
    feedback: Optional[str] = None

class UserStoryCollection(BaseModel):
    feature_request: Optional[str] = ""
    stories: Optional[List[UserStory]] = None

class Grade(BaseModel):
    """Binary score for relevance check."""

    binary_score: str = Field(description="Relevance score 'yes' or 'no'", default="no")