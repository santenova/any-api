# agents/validator_agent.py
import json
import networkx as nx
from pydantic import BaseModel
from .agent_base import AgentBase
# Define the schema for the response
class ModelInfo(BaseModel):
  name: str
  keywords: list[str]



class ModelList(BaseModel):
  results: list[ModelInfo]


class ConceptAgent(AgentBase):
    def __init__(self, max_retries=2, verbose=True, model_name = ""):
        super().__init__(name="ConceptAgent", max_retries=max_retries, verbose=verbose)
        self.seen_concepts = set()
        self.last_added = None
        self.current_concept = None
        self.graph = nx.DiGraph()
        self.model = None


    def execute(self, text, depth=1):
      full_path=""
      messages = [
        {"role": "system", "content": f"You are an AI assistant that creates concept lists from {text}."}
      ]

      concepts = self.call_llama(messages, max_tokens=1000,format=ModelList.model_json_schema())
      concepts = json.loads(concepts)
      return concepts
