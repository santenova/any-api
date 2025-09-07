# agents/summarize_agent.py

from .agent_base import AgentBase

class OutlineTool(AgentBase):
    def __init__(self, max_retries=2, verbose=True):
        super().__init__(name="OutlineTool", max_retries=max_retries, verbose=verbose)

    def execute(self, text):
        messages = [
            {"role": "system", "content": "You are an AI assistant that extract the Outline of a text."},
            {
                "role": "user",
                "content": (
                    "Please extract the Outline of the following text:\n\n"
                    f"{text}\n\nOutline:"
                )
            }
        ]
        outline = self.call_llama(messages, max_tokens=512)
        return outline
