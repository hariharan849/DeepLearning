from pydantic import BaseModel, Field
from typing import List, Dict, Optional, TypedDict
from state import design_doc


class Code(BaseModel):
    """Schema for code solutions to questions about LCEL."""
    prefix: str = Field(description="Description of the problem and approach")
    imports: str = Field(description="Code block import statements")
    code: str = Field(description="Code block not including import statements")

class CodeResponse(BaseModel):
    code: str
    reviewed_code: Optional[str] = None
    auto_review: Optional[str] = None
    security_issues: Optional[str] = None
    security_feedback: Optional[str] = None
    qa_issues: Optional[str] = None
    qa_feedback: Optional[str] = None
    unit_test: Optional[str] = None
    human_approved: int = 0
    feedback: Optional[str] = None

class DeveloperDocState(BaseModel):
    designer: design_doc.DesignDocState
    code: Optional[CodeResponse] = None

