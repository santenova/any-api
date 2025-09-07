# agents/write_book_agent.py
# agents/validator_agent.py
import json
from pydantic import BaseModel
from .agent_base import AgentBase


class BookInfo(BaseModel):
  name: str
  keywords: list[str]
  text: list[str]

class BookList(BaseModel):
  results: list[BookInfo]

class WriteBookTool(AgentBase):
    def __init__(self,max_retries=2, verbose=True):
        super().__init__(name="WriteBookTool", max_retries=max_retries, verbose=verbose)

    def execute(self, topic, outline=None):
        system_message = "You are an expert academic writer."
        user_content = f"Write a research book on the following topic:\nTopic: {topic}\n\n"
        if outline:
            user_content += f"Outline:\n{outline}\n\n"
        user_content += "Book:\n"
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_content}
        ]

        #book = self.call_llama(messages, max_tokens=1000,format=BookList.model_json_schema())
        book = self.call_llama(messages, max_tokens=1024)
        return book
