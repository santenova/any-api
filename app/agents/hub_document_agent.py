from .agent_base import AgentBase

class HubDocumentAgent(AgentBase):
    def __init__(self, max_retries=2, verbose=True):
        super().__init__(name="HubDocumentAgent", max_retries=max_retries, verbose=verbose)

    def execute(self, repo_name):
        system_message = "You are an expert software documentation and repository analysis agent."
        user_content = f"Analyze and generate comprehensive documentation for the repository: {repo_name}\n\n"
        user_content += "Provide details about:\n"
        user_content += "1. Repository structure\n"
        user_content += "2. Key components and their purposes\n"
        user_content += "3. Setup and installation instructions\n"
        user_content += "4. Key dependencies\n"
        user_content += "5. Any special configuration or usage notes\n"

        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_content}
        ]

        #documentation = self.call_llama(messages, max_tokens=1000)
        return messages

