from typing import Optional
from state import developer_doc
from pydantic import BaseModel


class SecurityState(BaseModel):
    developer: developer_doc.DeveloperDocState
    security_issues: Optional[str] = None
    reviewed_security_issues: Optional[str] = None
    human_approved: int = 0
    feedback: Optional[str] = None
