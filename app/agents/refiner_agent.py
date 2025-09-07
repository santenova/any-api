# agents/refiner_agent.py

from .agent_base import AgentBase

class RefinerAgent(AgentBase):
    def __init__(self, max_retries=2, verbose=True):
        super().__init__(name="RefinerAgent", max_retries=max_retries, verbose=verbose)

    def execute(self, draft):
        #draft = draft.decode("utf8")


        messages = [
            {"role": "system", "content": "You are an expert editor who refines and enhances research books for clarity, coherence, and academic quality."},
            {"role": "user", "content": f"Please refine the following research book draft to improve its language, coherence, and overall quality never shorten the word number:\n\n{draft}"}
        ]
        refined_book = self.call_llama(
            messages=messages,
            temperature=0.5,
            max_tokens=2048
        )
        return refined_book
