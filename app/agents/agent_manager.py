import random
from .summarize_tool import SummarizeTool
from .write_book_tool import WriteBookTool
from .sanitize_data_tool import SanitizeDataTool
from .summarize_validator_agent import SummarizeValidatorAgent
from .write_book_validator_agent import WriteBookValidatorAgent
from .sanitize_data_validator_agent import SanitizeDataValidatorAgent
from .refiner_agent import RefinerAgent
from .validator_agent import ValidatorAgent
from .concept_agent import ConceptAgent
from .models_agent import ModelsAgent

# New imports for hub command agents
from .hub_base_agent import HubBaseAgent
from .hub_document_agent import HubDocumentAgent
from .hub_tests_agent import HubTestsAgent
from .hub_analyse_agent import HubAnalyseAgent
from .hub_tests_agent import HubTestsAgent
from .hub_review_agent import HubReviewAgent
from .hub_assign_agent import HubAssignAgent
from .hub_products_agent import HubProductsAgent
from .hub_contributions_agent import HubContributionsAgent
from .extract_outline_tool import OutlineTool

class AgentManager:
    def __init__(self, max_retries=2, verbose=True):
        """
        self.models = ModelsAgent( max_retries=max_retries, verbose=verbose).execute()

        model_names = self.models['small']


        random.shuffle(model_names)
        """
        self.model = "gemma3:12b"
        #model_names[0]

        #print(self.model)


        self.agents = {
            "summarize": SummarizeTool( max_retries=max_retries, verbose=verbose),
            "write_book": WriteBookTool( max_retries=max_retries, verbose=verbose),
            "sanitize_data": SanitizeDataTool( max_retries=max_retries, verbose=verbose),
            "summarize_validator": SummarizeValidatorAgent( max_retries=max_retries, verbose=verbose),
            "outline_tool": OutlineTool( max_retries=max_retries, verbose=verbose),
            "write_book_validator": WriteBookValidatorAgent( max_retries=max_retries, verbose=verbose),
            "sanitize_data_validator": SanitizeDataValidatorAgent( max_retries=max_retries, verbose=verbose),
            "refiner": RefinerAgent( max_retries=max_retries, verbose=verbose),
            "validator": ValidatorAgent( max_retries=max_retries, verbose=verbose),
            "concept_tool": ConceptAgent( max_retries=max_retries, verbose=verbose),
            "models": ModelsAgent( max_retries=max_retries, verbose=verbose),
            # New hub command agents
            "hub_base": HubBaseAgent( max_retries=max_retries, verbose=verbose),
            "hub_document": HubDocumentAgent( max_retries=max_retries, verbose=verbose),
            "hub_review": HubReviewAgent( max_retries=max_retries, verbose=verbose),
            "hub_tests": HubTestsAgent( max_retries=max_retries, verbose=verbose),
            "hub_analyse": HubAnalyseAgent( max_retries=max_retries, verbose=verbose),
            "hub_assign": HubAssignAgent( max_retries=max_retries, verbose=verbose),
            "hub_contributions": HubContributionsAgent( max_retries=max_retries, verbose=verbose),
            "hub_products": HubProductsAgent( max_retries=max_retries, verbose=verbose)
        }

    def get_agent(self, agent_name, model_name=None):


        if model_name == None:
          model_name=self.model

        agent = self.agents.get(agent_name)

        if not agent:
            raise ValueError(f"Agent '{agent_name}' not found.")

        agent=agent.load(model_name)
        return agent


    def get_all_agents(self):
        agents = self.agents
        return agents
