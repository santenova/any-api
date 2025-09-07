from .agent_base import AgentBase

class HubReviewAgent(AgentBase):
    def __init__(self, max_retries=2, verbose=True):
        super().__init__(name="HubReviewAgent", max_retries=max_retries, verbose=verbose)

    def execute(self, repo_name):
        system_message = "You are an expert software testing and quality assurance agent."
        user_content = f"Analyze test coverage and quality for the repository: {repo_name}\n\n"
        user_content += "Provide insights on:\n"
        user_content += "1. Current test coverage\n"
        user_content += "2. Recommended additional test cases\n"
        user_content += "3. Potential areas of improvement\n"
        user_content += "4. Testing strategy and best practices\n"

        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_content}
        ]

        #contribution_analysis = self.call_llama(messages, max_tokens=1000)
        return messages
