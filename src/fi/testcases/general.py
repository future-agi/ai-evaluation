import json
from typing import List, Optional, Union

from pydantic import BaseModel


class TestCase(BaseModel):
    text: Optional[str] = None
    document: Optional[str] = None
    input: Optional[str] = None
    output: Optional[str] = None
    prompt: Optional[str] = None
    criteria: Optional[str] = None
    actual_json: Optional[dict] = None
    expected_json: Optional[dict] = None
    expected_text: Optional[str] = None
    expected_response: Optional[str] = None
    query: Optional[str] = None
    response: Optional[str] = None
    context: Union[List[str], str] = None

    def model_post_init(self, __context):
        if self.actual_json is not None:
            self.actual_json = json.loads(self.actual_json)
        if self.expected_json is not None:
            self.expected_json = json.loads(self.expected_json)
