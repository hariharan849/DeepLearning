from typing import Optional, TypedDict
from state import developer_doc
from pydantic import BaseModel

class UnitTestState(BaseModel):
    developer: developer_doc.DeveloperDocState
    unit_test_code: Optional[str] = None
    reviewed_unit_test_issues: Optional[str] = None
    human_approved: int = 0
    feedback: Optional[str] = None