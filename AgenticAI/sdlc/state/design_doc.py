from pydantic import BaseModel
from typing import Optional, TypedDict, List
from state.product_owner import UserStory


class DesignDocument(BaseModel):
    doc: str
    reviewed_doc: Optional[str] = None
    human_approved: int = 0
    feedback: Optional[str] = None

class DesignDocState(BaseModel):
    story: UserStory
    design_docs: Optional[DesignDocument] = None

