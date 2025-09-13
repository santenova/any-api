# agents/write_book_agent.py
# agents/validator_agent.py
import json, time
from pydantic import BaseModel
from .agent_base import AgentBase


class BookInfo(BaseModel):
  name: str
  keywords: list[str]
  text: list[str]
  integrety:int


class BookList(BaseModel):
  results: list[BookInfo]

class WriteBookTool(AgentBase):
    def __init__(self,max_retries=2, verbose=True):
        super().__init__(name="WriteBookTool", max_retries=max_retries, verbose=verbose)



    def validate_book(self, topic, book):
        system_message = ""
        user_content = (
            "Provide a brief analysis and sore the book on a scale of 1 to 100, where 100% indicates excellent quality."
            f"Topic:\n{topic}\n\n"
            f"Book:\n{book}\n\n"
            "Validation:"
        )
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_content}
        ]
        validation = self.call_llama(messages, max_tokens=512)
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

    def conceptual(self, text):
      messages = [
            {"role": "system", "content": "You are an AI assistant that creates concept lists from text."},
            {
                "role": "user",
                "content": f"""
Starting with the concept: "{text}", generate {5} to 30, of the most close related instances to our Starting concept.
#
Guidelines:
1. Seek maximum intellectual diversity - span across domains like science, art, philosophy, technology, culture, etc.
2. Each concept should be expressed in 1-5 words (shorter is better)
3. Avoid obvious associations - prefer surprising or thought-provoking connections
4. Consider how your suggested concepts relate to BOTH:
   - The immediate parent concept "{text}"
5. Consider these different types of relationships:
   - Metaphorical parallels
   - Contrasting opposites
   - Historical connections
   - Philosophical implications
   - Cross-disciplinary applications

Avoid any concepts already in the path. Be creative but maintain meaningful connections.

Return ONLY a JSON array of strings, with no explanation or additional text.
Example: ["Related concept 1", "Related concept 2", "Related concept 3", "Related concept 4","Related concept 5", "Related concept 6", "Related concept 7", "Related concept 8"]
        """
            }
        ]


      summary = self.call_llama(messages, max_tokens=512)
      return summary

    def create_keywords(self,user_query):
      # Create a comprehensive prompt
        prompt = f"""Create Keywords for this question

Question: {user_query}

Return ONLY a JSON array of strings, with no explanation or additional text.
Example: ["Keyword 1", "Keyword 2", "Keyword 3", "Keyword 4","Keyword 5", "Keyword 6", "Keyword 7", "Keyword 8"]
"""
        messages = [
            {"role": "system", "content": "You are an AI assistant that creates keywords from text."},
            {"role": "user", "content": prompt}
        ]
        keywords = self.call_llama(
            messages=messages,
            temperature=0.3,         # Lower temperature for more deterministic output
            max_tokens=500
        )

        print(keywords.replace("```json","").replace("```",""))
        return keywords


    def execute(self, topic, outline=None):
        system_message = "You are an expert writer."
        user_content = f"Write a research book on the following topic:\nTopic: {topic}\n\n"

        if outline is None:
            keywords = self.create_keywords(topic).replace("json```","").replace("```","")
            outline = self.conceptual(topic).replace("json```","").replace("```","")
            user_content += f"Outline:\n{outline}\nKeywors:{keywords}\n\n"


        user_content += "Book:\n"
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_content}
        ]



        book = self.call_llama(messages, max_tokens=1024)

        time.sleep(1)

        validation = self.validate_book(topic,book)
        print(validation)

        book_summary = self.summarize(book)
        time.sleep(1000)
        book_outline = self.outline(book)
        book_refine = self.refine(book)
        time.sleep(1000)
        book_validated = self.validate_book(user_content,book)
        time.sleep(1000)
        summary_validated = self.validate_summerize(book,book_summary)
        output_file = f"data/{self.model}_book.txt"

        with open(output_file.replace(".txt",".json"), 'w', encoding='utf-8') as f:
            json.dump([messages,output_file,book,outline], f, ensure_ascii=False, indent=4)



        return {"topic":topic,"instructions":messages,"model":self.model,"book":book,"validation":validation,"file":output_file}
