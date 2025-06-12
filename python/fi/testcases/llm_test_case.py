from typing import List, Optional, Union

from pydantic import BaseModel


class LLMTestCase(BaseModel):
    query: str
    response: str
    context: Optional[Union[str, List[str]]] = None
    expected_response: Optional[str] = None
