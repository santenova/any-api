# agents/write_book_agent.py
# agents/validator_agent.py
import json, time
from pydantic import BaseModel
from .agent_base import AgentBase


class BookInfo(BaseModel):
  name: str
  keywords: list[str]
  text: list[str]

class BookList(BaseModel):
  results: list[BookInfo]

class WriteBookTool(AgentBase):
    def __init__(self,max_retries=2, verbose=True):
        super().__init__(name="WriteBookTool", max_retries=max_retries, verbose=verbose)



    def validate_book(self, topic, book):
        text = f"""
               Given the topic and the research book below, assess whether the book comprehensively covers the topic, follows a logical structure, and maintains academic standards.
               Provide a brief analysis and rate the book on a scale of 1 to 100, where 100 indicates excellent quality.
               Topic: {topic}
               Book:\n{book}
               Validation:
               """

        messages = [
            {"role": "system", "content": "You are an AI assistant that validates research books for accuracy, completeness, and adherence to academic standards."},
            {"role": "user", "content": text}
        ]
        validation = self.call_llama(
            messages=messages,
            temperature=0.3,         # Lower temperature for more deterministic output
            max_tokens=500
        )
        return validation


    def validate_summerize(self, original_text, summary):
        system_message = ""
        user_content = (
            "Given the original text and its summary, assess whether the summary accurately and concisely captures the key points of the original text.\n"
            "Provide a brief analysis and rate the summary on a scale of 1 to 100, where 100 indicates excellent quality explain how you got there.\n\n"
            f"Original Text:\n{original_text}\n\n"
            f"Summary:\n{summary}\n\n"
            "Validation:"
        )
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_content}
        ]
        validation = self.call_llama(messages, max_tokens=512)
        return validation


    def validate_sanitized(self, original_data, sanitized_data):
        system_message = "You are an AI assistant that validates the sanitization of medical data by checking for the removal of Protected Health Information (PHI)."
        user_content = (
            "Given the original data and the sanitized data, verify that all PHI has been removed.\n"
            "List any remaining PHI in the sanitized data and rate the sanitization process on a scale of 1 to 100, where 100 indicates complete sanitization.\n\n"
            f"Original Data:\n{original_data}\n\n"
            f"Sanitized Data:\n{sanitized_data}\n\n"
            "Validation:"
        )
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_content}
        ]
        validation = self.call_llama(messages, max_tokens=512)
        return validation




    def outline(self, text):
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


    def refine(self, draft):
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

    def sanitize(self, medical_data):
        messages = [
            {"role": "system", "content": "You are an AI assistant that sanitizes medical data by removing Protected Health Information (PHI)."},
            {
                "role": "user",
                "content": (
                    "Remove all PHI from the following data:\n\n"
                    f"{medical_data}\n\nSanitized Data:"
                )
            }
        ]
        sanitized_data = self.call_llama(messages, max_tokens=500)
        return sanitized_data

    def summarize(self, text):
        messages = [
            {"role": "system", "content": "You are an AI assistant that summarizes text."},
            {
                "role": "user",
                "content": (
                    "Please provide a concise summary of the following text:\n\n"
                    f"{text}\n\nSummary:"
                )
            }
        ]
        summary = self.call_llama(messages, max_tokens=512)
        return summary



    def execute(self, topic, outline=None):
        system_message = "You are an expert academic writer."
        user_content = f"Write a research book on the following topic:\nTopic: {topic}\n\n"
        if outline:
            user_content += f"Outline:\n{outline}\n\n"
        user_content += "Book:\n"
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_content}
        ]

        #book = self.call_llama(messages, max_tokens=1000,format=BookList.model_json_schema())
        book = self.call_llama(messages, max_tokens=1024)
        time.sleep(1000)
        book_summary = self.summarize(book)
        time.sleep(1000)
        book_outline = self.outline(book)
        book_refine = self.refine(book)
        time.sleep(1000)
        book_validated = self.validate_book(user_content,book)
        time.sleep(1000)
        summary_validated = self.validate_summerize(book,book_summary)

        return book
