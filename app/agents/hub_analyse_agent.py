from .agent_base import AgentBase

class HubAnalyseAgent(AgentBase):
    def __init__(self, max_retries=2, verbose=True):
        super().__init__(name="HubAnalyseAgent", max_retries=max_retries, verbose=verbose)

    def execute(self, repo_name):
        system_message = "You are an expert software architecture and code analysis agent."
        user_content = f"Perform comprehensive analysis of the repository: {repo_name}\n\n"
        user_content += "Analyze and provide insights on:\n"
        user_content += "1. Code quality and complexity\n"
        user_content += "2. Architectural patterns and design\n"
        user_content += "3. Potential performance bottlenecks\n"
        user_content += "4. Security considerations\n"
        user_content += "5. Recommendations for improvement\n"

        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_content}
        ]

        #analysis_report = self.call_llama(messages, max_tokens=1000)
        return messages
