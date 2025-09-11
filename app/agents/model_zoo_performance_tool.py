import os
import asyncio
import requests, time, random
from .agent_base import AgentBase
from dotenv import load_dotenv


load_dotenv()


TRIPLE_VALIDATION_PROMPT = """
You are a triple validator for a personal knowledge graph.

Given an utterance that a user spoke to a voice assistant and a candidate triple, your task is to validate the triple

Utterances about the user usually have the form of "I am ...." or "My ..."

Utterances about the assistant usually have the form of "You are ...." or "Your ..."

Knowledge about the broader world should be discarded, you are only interested in personal information about the user or the voice assistant

Each triple is in the format:
(subject, predicate, object)

Only return 'True' if:
- The subject is 'self' (the assistant) or 'user' (the user)
- The triple is about user or assistant personal information
- The triple is factually plausible and makes sense
- The triple DOES NOT contradict the utterance

Otherwise, return 'False'.

Examples of valid triples:
"my favorite color is green" - ("user", "schema:favoriteColor", "green")
"your favorite color is blue" - ("self", "schema:favoriteColor", "blue")

Examples of invalid triples:
"my favorite color is green" - ("user", "schema:favoriteColor", "red")
"I love the color green" - ("self", "schema:favoriteColor", "green")
"your favorite color is blue" - ("user", "schema:favoriteColor", "blue")

YOU MUST answer with only one word: True or False.

The user said: "{utterance}"

Candidate triple: {triple}
"""

VALIDATION_PROMPT = """
Select the Model with higest score

data: {json}
"""

OLLAMA_BASE = os.getenv('OLLAMA_BASE')
DEFAULT_MODEL = os.getenv('DEFAULT_MODEL')

class ModelValidator(AgentBase):
    def __init__(self, max_retries=2, verbose=True):
        super().__init__(name="ModelValidator", max_retries=max_retries, verbose=verbose)

    def get_models(self):

        response = requests.request("GET",f"{OLLAMA_BASE}/api/tags")
        response.raise_for_status()
        result=[]
        models = response.json()
        for model in models['models']:
          result.append(model["name"])


        self.model_name = result[0]
        self.base_url = OLLAMA_BASE

        return result

    def validate_triple_ollama(self,utterance, triple, model="gemma2"):
        prompt = TRIPLE_VALIDATION_PROMPT.format(utterance=utterance,
                                                 triple=repr(triple))
        response = requests.post(
            f"{OLLAMA_BASE}//api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.0,
                    "num_predict": 1,
                    "stop": ["\n"]
                }
            }
        )

        result = response.json()["response"].strip()
        return result.lower() == "true"


    def validate_results(self,utterance):

        sorted_returns = dict(sorted(utterance.items(), key=lambda x: x[1]['score'], reverse=True))
        for s in sorted_returns:
          return {s:sorted_returns[s]['score']}


    def main(self):

        valid = [
            ("my favorite color is green", ("user", "schema:favoriteColor", "green")),
            ("your favorite color is green", ("self", "schema:favoriteColor", "green"))
        ]
        invalid = [
            ("my favorite color is green", ("user", "schema:favoriteColor", "red")),
            ("my favorite color is green", ("self", "schema:favoriteColor", "green")),
            ("your favorite color is green", ("user", "schema:favoriteColor", "green")),
            ("your favorite color is green", ("self", "schema:favoriteColor", "red")),
        ]

        valid += [
            ("I am not interested in sports", ("user", "schema:interestInSports", "no")),
            ("You are not programmed to have preferences", ("self", "schema:preferences", "none"))
        ]
        invalid += [
            ("I am not interested in sports", ("self", "schema:interestInSports", "no")),
            ("You are not programmed to have preferences", ("user", "schema:preferences", "none"))
        ]

        valid += [
            ("I am a software engineer", ("user", "schema:occupation", "software engineer")),
            ("You are an assistant", ("self", "schema:occupation", "assistant")),
            ("my name is John", ("user", "schema:name", "John")),
            ("your name is Ollama", ("self", "schema:name", "Ollama")),
            ("I live in New York", ("user", "schema:homeLocation", "New York")),
            ("You are located in the cloud", ("self", "schema:homeLocation", "cloud")),
            ("my birthday is January 1st", ("user", "schema:birthDate", "January 1st")),
            ("your favorite food is pizza", ("self", "schema:favoriteFood", "pizza"))
        ]
        invalid += [
            ("I am a software engineer", ("self", "schema:occupation", "software engineer")),
            ("You are an assistant", ("user", "schema:occupation", "assistant")),
            ("my name is John", ("self", "schema:name", "John")),
            ("your name is Ollama", ("user", "schema:name", "Ollama")),
            ("I live in New York", ("self", "schema:homeLocation", "New York")),
            ("You are located in the cloud", ("user", "schema:homeLocation", "cloud")),
            ("my birthday is January 1st", ("self", "schema:birthDate", "January 1st")),
            ("your favorite food is pizza", ("user", "schema:favoriteFood", "pizza"))
        ]
        valid += [
            ("I am a doctor", ("user", "schema:occupation", "doctor")),
            ("You are a virtual assistant", ("self", "schema:occupation", "virtual assistant")),
            ("My name is Alice", ("user", "schema:name", "Alice")),
            ("Your name is Siri", ("self", "schema:name", "Siri")),
            ("I reside in London", ("user", "schema:homeLocation", "London")),
            ("You are based in the cloud", ("self", "schema:homeLocation", "cloud")),
            ("My birthdate is December 25th", ("user", "schema:birthDate", "December 25th")),
            ("Your favorite drink is coffee", ("self", "schema:favoriteFood", "coffee")),
            ("I am a writer", ("user", "schema:occupation", "writer")),
            ("Your favorite color is red", ("self", "schema:favoriteColor", "red"))
        ]
        invalid += [
            ("I am a doctor", ("self", "schema:occupation", "doctor")),
            ("You are a virtual assistant", ("user", "schema:occupation", "virtual assistant")),
            ("My name is Alice", ("self", "schema:name", "Alice")),
            ("Your name is Siri", ("user", "schema:name", "Siri")),
            ("I reside in London", ("self", "schema:homeLocation", "London")),
            ("You are based in the cloud", ("user", "schema:homeLocation", "cloud")),
            ("My birthdate is December 25th", ("self", "schema:birthDate", "December 25th")),
            ("Your favorite drink is coffee", ("user", "schema:favoriteFood", "coffee")),
            ("I am a writer", ("self", "schema:occupation", "writer")),
            ("Your favorite color is red", ("user", "schema:favoriteColor", "red"))
        ]
        valid += [
            ("I have been working as an engineer for 10 years", ("user", "schema:occupation", "engineer")),
            ("You have been active since 2020", ("self", "schema:activeSince", "2020")),
            ("My favorite sport is basketball", ("user", "schema:favoriteSport", "basketball")),
            ("Your primary role is to assist users", ("self", "schema:primaryRole", "assist users")),
            ("I celebrate my birthday every year on October 15th", ("user", "schema:birthDate", "October 15th"))
        ]
        invalid += [
            ("I have been working as an engineer for 10 years", ("self", "schema:occupation", "engineer")),
            ("You have been active since 2020", ("user", "schema:activeSince", "2020")),
            ("My favorite sport is basketball", ("self", "schema:favoriteSport", "basketball")),
            ("Your primary role is to assist users", ("user", "schema:primaryRole", "assist users")),
            ("I celebrate my birthday every year on October 15th", ("self", "schema:birthDate", "October 15th"))
        ]
        valid += [
            ("I love coding", ("user", "schema:favoriteHobby", "coding")),
            ("You help people with their tasks", ("self", "schema:primaryRole", "help people")),
            ("My city of residence is Tokyo", ("user", "schema:homeLocation", "Tokyo")),
            ("Your preferred language is Python", ("self", "schema:preferredLanguage", "Python"))
        ]
        invalid += [
            ("I love coding", ("self", "schema:favoriteHobby", "coding")),
            ("You help people with their tasks", ("user", "schema:primaryRole", "help people")),
            ("My city of residence is Tokyo", ("self", "schema:homeLocation", "Tokyo")),
            ("Your preferred language is Python", ("user", "schema:preferredLanguage", "Python")),
        ]


        MODELS = self.get_models()
        random.shuffle(MODELS)
        returns = {}

        for model in MODELS:

          try:

            tic = time.perf_counter()
            print("\n#######################")
            print(f"## Testing model: {model}")
            print()
            correct = 0
            wrong = 0
            fn = 0

            fp = 0
            for utt, triple in valid:
                if self.validate_triple_ollama(utt, triple, model):
                    correct += 1
                    print("✅ Correctly concluded a Invalid test:\t\t",utt, triple)

                else:
                    wrong += 1
                    fn += 1
                    print("❌ Incorrectly concluded as Invalid test:\t\t",utt, triple)

            for utt, triple in invalid:
                if self.validate_triple_ollama(utt, triple, model):
                    wrong += 1
                    fp += 1
                    print("❌ Incorrectly concluded as Valid test:\t\t",utt, triple)
                else:
                    print("✅ Correctly concluded a Valid test:\t\t",utt, triple)
                    correct += 1

            toc = time.perf_counter()
            total = len(valid) + len(invalid)
            tpq = int(toc - tic) / total
            score = f"{correct / total:0.3f}"
            returns[model]={"model":model,"score":score,"correct":correct,"wrong":wrong,"false positives":fp,"false negatives":fn,"durration":f" {toc - tic:0.1f} seconds","time pro question":tpq}
            bestscore=self.validate_results(returns,MODELS[0])
            print(f"\n\nTesting time: {toc - tic:0.4f} seconds")
            print(f"Model:{model} {score}")
            print("Total correct predictions:", correct)
            print("Total wrong predictions:", wrong)
            print("     - false positives:",  fp)
            print("     - false negatives:",  fn)
            print(f"     - durration:{toc - tic:0.1f}")
            print(f"     - best model:{bestscore}")
            #print(returns[model])
          except:
            pass



        print(self.validate_results(returns))




        return returns


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

        ret = self.main()
        bestscore=self.validate_results(ret)
        return {"best":bestscore,"all":ret}



"""
class ModelValidator:
    def __init__(self, max_retries=2, verbose=True):
        super().__init__(name="ModelValidator", max_retries=max_retries, verbose=verbose)

    def get_models(self):

        response = requests.request("GET",f"{OLLAMA_BASE}/api/tags")
        response.raise_for_status()
        result=[]
        models = response.json()
        for model in models['models']:
          result.append(model["name"])


        self.model_name = result[0]
        self.base_url = OLLAMA_BASE

        return result




    def validate_triple_ollama(self,utterance, triple, model="gemma2"):
        prompt = TRIPLE_VALIDATION_PROMPT.format(utterance=utterance,
                                                 triple=repr(triple))
        response = requests.post(
            f"{OLLAMA_BASE}//api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.0,
                    "num_predict": 1,
                    "stop": ["\n"]
                }
            }
        )

        result = response.json()["response"].strip()
        return result.lower() == "true"


    def validate_results(self,utterance, model="gemma2"):

        sorted_returns = dict(sorted(utterance.items(), key=lambda x: x[1]['score'], reverse=True))
        for s in sorted_returns:
          return {s:sorted_returns[s]['score']}




    def main(self):

        valid = [
            ("my favorite color is green", ("user", "schema:favoriteColor", "green")),
            ("your favorite color is green", ("self", "schema:favoriteColor", "green"))
        ]
        invalid = [
            ("my favorite color is green", ("user", "schema:favoriteColor", "red")),
            ("my favorite color is green", ("self", "schema:favoriteColor", "green")),
            ("your favorite color is green", ("user", "schema:favoriteColor", "green")),
            ("your favorite color is green", ("self", "schema:favoriteColor", "red")),
        ]

        valid += [
            ("I am not interested in sports", ("user", "schema:interestInSports", "no")),
            ("You are not programmed to have preferences", ("self", "schema:preferences", "none"))
        ]
        invalid += [
            ("I am not interested in sports", ("self", "schema:interestInSports", "no")),
            ("You are not programmed to have preferences", ("user", "schema:preferences", "none"))
        ]

        valid += [
            ("I am a software engineer", ("user", "schema:occupation", "software engineer")),
            ("You are an assistant", ("self", "schema:occupation", "assistant")),
            ("my name is John", ("user", "schema:name", "John")),
            ("your name is Ollama", ("self", "schema:name", "Ollama")),
            ("I live in New York", ("user", "schema:homeLocation", "New York")),
            ("You are located in the cloud", ("self", "schema:homeLocation", "cloud")),
            ("my birthday is January 1st", ("user", "schema:birthDate", "January 1st")),
            ("your favorite food is pizza", ("self", "schema:favoriteFood", "pizza"))
        ]
        invalid += [
            ("I am a software engineer", ("self", "schema:occupation", "software engineer")),
            ("You are an assistant", ("user", "schema:occupation", "assistant")),
            ("my name is John", ("self", "schema:name", "John")),
            ("your name is Ollama", ("user", "schema:name", "Ollama")),
            ("I live in New York", ("self", "schema:homeLocation", "New York")),
            ("You are located in the cloud", ("user", "schema:homeLocation", "cloud")),
            ("my birthday is January 1st", ("self", "schema:birthDate", "January 1st")),
            ("your favorite food is pizza", ("user", "schema:favoriteFood", "pizza"))
        ]
        valid += [
            ("I am a doctor", ("user", "schema:occupation", "doctor")),
            ("You are a virtual assistant", ("self", "schema:occupation", "virtual assistant")),
            ("My name is Alice", ("user", "schema:name", "Alice")),
            ("Your name is Siri", ("self", "schema:name", "Siri")),
            ("I reside in London", ("user", "schema:homeLocation", "London")),
            ("You are based in the cloud", ("self", "schema:homeLocation", "cloud")),
            ("My birthdate is December 25th", ("user", "schema:birthDate", "December 25th")),
            ("Your favorite drink is coffee", ("self", "schema:favoriteFood", "coffee")),
            ("I am a writer", ("user", "schema:occupation", "writer")),
            ("Your favorite color is red", ("self", "schema:favoriteColor", "red"))
        ]
        invalid += [
            ("I am a doctor", ("self", "schema:occupation", "doctor")),
            ("You are a virtual assistant", ("user", "schema:occupation", "virtual assistant")),
            ("My name is Alice", ("self", "schema:name", "Alice")),
            ("Your name is Siri", ("user", "schema:name", "Siri")),
            ("I reside in London", ("self", "schema:homeLocation", "London")),
            ("You are based in the cloud", ("user", "schema:homeLocation", "cloud")),
            ("My birthdate is December 25th", ("self", "schema:birthDate", "December 25th")),
            ("Your favorite drink is coffee", ("user", "schema:favoriteFood", "coffee")),
            ("I am a writer", ("self", "schema:occupation", "writer")),
            ("Your favorite color is red", ("user", "schema:favoriteColor", "red"))
        ]
        valid += [
            ("I have been working as an engineer for 10 years", ("user", "schema:occupation", "engineer")),
            ("You have been active since 2020", ("self", "schema:activeSince", "2020")),
            ("My favorite sport is basketball", ("user", "schema:favoriteSport", "basketball")),
            ("Your primary role is to assist users", ("self", "schema:primaryRole", "assist users")),
            ("I celebrate my birthday every year on October 15th", ("user", "schema:birthDate", "October 15th"))
        ]
        invalid += [
            ("I have been working as an engineer for 10 years", ("self", "schema:occupation", "engineer")),
            ("You have been active since 2020", ("user", "schema:activeSince", "2020")),
            ("My favorite sport is basketball", ("self", "schema:favoriteSport", "basketball")),
            ("Your primary role is to assist users", ("user", "schema:primaryRole", "assist users")),
            ("I celebrate my birthday every year on October 15th", ("self", "schema:birthDate", "October 15th"))
        ]
        valid += [
            ("I love coding", ("user", "schema:favoriteHobby", "coding")),
            ("You help people with their tasks", ("self", "schema:primaryRole", "help people")),
            ("My city of residence is Tokyo", ("user", "schema:homeLocation", "Tokyo")),
            ("Your preferred language is Python", ("self", "schema:preferredLanguage", "Python"))
        ]
        invalid += [
            ("I love coding", ("self", "schema:favoriteHobby", "coding")),
            ("You help people with their tasks", ("user", "schema:primaryRole", "help people")),
            ("My city of residence is Tokyo", ("self", "schema:homeLocation", "Tokyo")),
            ("Your preferred language is Python", ("user", "schema:preferredLanguage", "Python")),
        ]


        MODELS = self.get_models()
        random.shuffle(MODELS)
        returns = {}

        for model in MODELS:

          try:

            tic = time.perf_counter()
            print("\n#######################")
            print(f"## Testing model: {model}")
            print()
            correct = 0
            wrong = 0
            fn = 0

            fp = 0
            for utt, triple in valid:
                if self.validate_triple_ollama(utt, triple, model):
                    correct += 1
                    print("✅ Correctly concluded a Invalid test:\t\t",utt, triple)

                else:
                    wrong += 1
                    fn += 1
                    print("❌ Incorrectly concluded as Invalid test:\t\t",utt, triple)

            for utt, triple in invalid:
                if self.validate_triple_ollama(utt, triple, model):
                    wrong += 1
                    fp += 1
                    print("❌ Incorrectly concluded as Valid test:\t\t",utt, triple)
                else:
                    print("✅ Correctly concluded a Valid test:\t\t",utt, triple)
                    correct += 1

            toc = time.perf_counter()
            total = len(valid) + len(invalid)
            tpq = int(toc - tic) / total
            score = f"{correct / total:0.3f}"
            returns[model]={"model":model,"score":score,"correct":correct,"wrong":wrong,"false positives":fp,"false negatives":fn,"durration":f" {toc - tic:0.1f} seconds","time pro question":tpq}
            bestscore=self.validate_results(returns,MODELS[0])
            print(f"\n\nTesting time: {toc - tic:0.4f} seconds")
            print(f"Model:{model} {score}")
            print("Total correct predictions:", correct)
            print("Total wrong predictions:", wrong)
            print("     - false positives:",  fp)
            print("     - false negatives:",  fn)
            print(f"     - durration:{toc - tic:0.1f}")
            print(f"     - best model:{bestscore}")
            #print(returns[model])
          except:
            pass



        print(self.validate_results(returns,MODELS[0]))




        return returns


    def execute(self):
      print("main")
      ret = self.main()
      bestscore=self.validate_results(ret,MODELS[0])
      return {"best":bestscore,"all":ret}

"""
