from pydantic import BaseModel
from typing import Annotated
import operator

class AgentState(BaseModel):
    request: str
    web_response: str | None = None
    sf_response: list | None = None
    final_output: str | None = None