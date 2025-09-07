# agents/write_book_validator_agent.py

from .agent_base import AgentBase

class WriteBookValidatorAgent(AgentBase):
    def __init__(self, max_retries=2, verbose=True):
        super().__init__(name="WriteBookValidatorAgent", max_retries=max_retries, verbose=verbose)

    def execute(self, topic, book):
        system_message = "You are an AI assistant that validates research books."
        user_content = (
            "Given the topic and the book, assess whether the book comprehensively covers the topic, follows a logical structure, and maintains academic standards.\n"
            "Provide a brief analysis and rate the book on a scale of 1 to 100, where 100 indicates excellent quality.\n\n"
            f"Topic: {topic}\n\n"
            f"Book:\n{book}\n\n"
            "Validation:"
        )
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_content}
        ]
        validation = self.call_llama(messages, max_tokens=512)
        return validation
