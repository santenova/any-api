# agents/validator_agent.py

from .agent_base import AgentBase

class ValidatorAgent(AgentBase):
    def __init__(self,max_retries=2, verbose=True):
        super().__init__(name="ValidatorAgent", max_retries=max_retries, verbose=verbose)

    def execute(self, topic, book):
        text = """
               You are an AI assistant that validates research books for accuracy, completeness, and adherence to academic standards.
               Given the topic and the research book below, assess whether the book comprehensively covers the topic, follows a logical structure, and maintains academic standards.
               Provide a brief analysis and rate the book on a scale of 1 to 100, where 100 indicates excellent quality.
               Topic: {topic}
               Book:\n{book}
               Validation:
               """

        messages = [
            {"role": "system", "content": "You are an expert editor who refines and enhances research books for clarity, coherence, and academic quality."},
            {"role": "user", "content": text}
        ]
        validation = self.call_llama(
            messages=messages,
            temperature=0.3,         # Lower temperature for more deterministic output
            max_tokens=500
        )
        return validation
