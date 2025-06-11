from typing import Any, List

from pydantic import BaseModel

from fi.testcases.llm_test_case import LLMTestCase


class ConversationalTestCase(BaseModel):
    messages: List[LLMTestCase]

    def model_post_init(self, __context: Any) -> None:
        if len(self.messages) == 0:
            raise TypeError("'messages' must not be empty")

        copied_messages = []
        for message in self.messages:
            if not isinstance(message, LLMTestCase):
                raise TypeError("'messages' must be a list of `LLMTestCases`")
            else:
                copied_messages.append(str(message.query))
                copied_messages.append(str(message.response))
        self.messages = copied_messages
