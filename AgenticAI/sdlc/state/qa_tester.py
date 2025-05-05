from typing import Optional, TypedDict
from state import developer_doc
from pydantic import BaseModel

class QATestState(BaseModel):
    developer: developer_doc.DeveloperDocState
    qa_test_code: Optional[str] = None
    unit_test_code: Optional[str] = None
    security_issue: Optional[str] = None
    human_approved: int = 0
    feedback: Optional[str] = None