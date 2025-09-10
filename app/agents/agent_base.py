# agents/agent_base.py

import ollama
from abc import ABC, abstractmethod
from loguru import logger
import os

class AgentBase(ABC):
    def __init__(self, name, max_retries=2, verbose=True):
        self.name = name
        self.max_retries = max_retries
        self.verbose = verbose

    @abstractmethod
    def execute(self, *args, **kwargs):
        pass

    def load(self,model_name = 'qwen2.5-coder:0.5b'):
        self.model = model_name
        return self


    def call_llama(self, messages, temperature=0.7, max_tokens=150, format=None,model = 'qwen2.5-coder:0.5b'):
        """
        Calls the Llama model via Ollama and retrieves the response.

        Args:
            messages (list): A list of message dictionaries.
            temperature (float): Sampling temperature.
            max_tokens (int): Maximum number of tokens in the response.

        Returns:
            str: The content of the model's response.
        """
        retries = 0

        while retries < self.max_retries:
            try:
                if self.verbose:
                    logger.info(f"[{self.name}][{self.model}]  Sending messages to Ollama:")
                    for msg in messages:
                        logger.debug(f"  {msg['role']}: {msg['content']}")

                # Call the Ollama chat API
                if format is None:
                  response = ollama.chat(
                      model=self.model,  # Updated model name
                      messages=messages
                  )
                else:
                  response = ollama.chat(
                      model=self.model,  # Updated model name
                      messages=messages,
                      format=format
                  )

                # Parse the response to extract the text content
                reply = response['message']['content']

                if self.verbose:
                    logger.info(f"[{self.name}][{self.model}]  Received response: {reply}")

                return reply
            except Exception as e:
                retries += 1
                logger.error(f"[{self.name}][{self.model}]  Error during Ollama call: {e}. Retry {retries}/{self.max_retries}")
        raise Exception(f"[{self.name}][{self.model}]  Failed to get response from Ollama after {self.max_retries} retries.")




#AgentBase("gemma3:1b")
