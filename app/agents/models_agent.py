# agents/validator_agent.py
import networkx as nx

import pprint, re
from ollama import list
from ollama import ListResponse
from ollama import ShowResponse, show


from .agent_base import AgentBase

class ModelsAgent(AgentBase):
    def __init__(self, max_retries=2, verbose=True):
        super().__init__(name="ModelsAgent", max_retries=max_retries, verbose=verbose)
        self.seen_concepts = set()
        self.last_added = None
        self.current_concept = None
        self.graph = nx.DiGraph()
        self.model = None
        self.large = 10000


    def execute(self):
      full_path=""
      response: ListResponse = list()
      out={'code':[],'small':[]}
      for model in response.models:
        response: ShowResponse = show(model.model)
        mb = (model.size.real / 1024 / 1024)
        mb = int(mb)
        #print('Model Information:')
        #print('  Size (MB):', f'{(model.size.real / 1024 / 1024):.2f}')
        #print(f'Modified at:   {response.modified_at}')
        #print(f'Template:      {response.template}')
        #print(f'Modelfile:     {response.modelfile}')
        #print(f'License:       {response.license}')
        #print(f'Details:       {response.details}')
        #print(f'Model Info:    {response.modelinfo}')
        #print(f'Parameters:    {response.parameters}')
        #print(f'Capabilities:  {response.capabilities}')
        for c in response.capabilities:
          #print(c)
          if c not in out:
             out[c]=[]
             out[c].append(model.model)
          else:
            out[c].append(model.model)

        for r in ['code','dev']:
          embeds = re.findall(r, model.model, flags=re.IGNORECASE)
          if embeds:
            out['code'].append(model.model)

        for r in ['embed']:
          embeds = re.findall(r, model.model, flags=re.IGNORECASE)
          if embeds:
            continue
          else:
            if mb > self.large:
              continue
            else:
              out['small'].append(model.model)


      return out


