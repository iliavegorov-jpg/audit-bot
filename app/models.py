from pydantic import BaseModel, Field
from typing import List, Dict, Any, Union, Optional

class DictItem(BaseModel):
    id: str
    name: str
    description: Optional[str] = None
    description_short: Optional[str] = None
    risk_type: Optional[str] = None
    scenario_short: Optional[str] = None

class UserInput(BaseModel):
    problem_text: str
    process_object: str
    period: str
    participants_roles: Optional[str] = None
    what_violated: Optional[str] = None
    amounts_terms: Optional[str] = None
    documents: Optional[str] = None

class SelectedFromDict(BaseModel):
    primary_id: str
    alternatives: List[str] = Field(default_factory=list)
    confidence: float = 0.0
    rationale: str = ""

class SectionVariants(BaseModel):
    text: Optional[str] = None
    variants: Optional[Union[List[str], Dict[str, Any]]] = Field(
        default=None,
        description="Either list of variant strings OR structured object"
    )

class LLMResponse(BaseModel):
    selected: Dict[str, SelectedFromDict]
    sections: Dict[str, SectionVariants]
