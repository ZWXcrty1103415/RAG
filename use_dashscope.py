
"""
using qwen and gte-rerank model with dashscope sdk
"""

import os

import dashscope
from openai import OpenAI

os.environ['OPENAI_API_KEY'] = 'sk-2ca88ec1a4ca4b619ad06bae0df80bf6'
os.environ['DASHSCOPE_API_KEY'] = 'sk-2ca88ec1a4ca4b619ad06bae0df80bf6'
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

openai_base = OpenAI(
    api_key=os.environ['OPENAI_API_KEY'],
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)
def message_parser(messages):
    query_messages = []
    for message in messages:
        query_message = {
            "role":message[0],
            "content":message[1]
        }
        query_messages.append(query_message)
    return query_messages

def message_generate(prompt, inputs):
    messages = []
    for message, insert in zip(prompt, inputs):
        if not insert:
            continue
        messages.append((message[0], message[1].format(**insert)))
    return messages

class LLMClient:
    def __init__(self,  model: str, prompt = None, base = openai_base):
        self.prompt = prompt
        self.model = model
        self.base = base
    def invoke(self, inserts, **kwargs):
        if self.prompt:
            message = message_generate(self.prompt, inserts)
        else:
            message = inserts
        response = self.base.chat.completions.create(
            model = self.model,
            messages= message_parser(message),
            **kwargs
        )
        return response.model_dump()['choices'][0]["message"]["content"]
    def stream(self, inserts, **kwargs):
        if self.prompt:
            message = message_generate(self.prompt, inserts)
        else:
            message = inserts
        return self.base.chat.completions.create(
            model = self.model,
            stream = True,
            messages= message_parser(message),
            **kwargs
        )

class RerankClient:
    def __init__(self, model: str):
        self.model = model
    def invoke(self, query:str, documents: list[str], top_n : int = 8, **kwargs):
        resp = dashscope.TextReRank.call(
            model = self.model,
            query = query,
            documents = documents,
            top_n = top_n,
            api_key=os.environ['DASHSCOPE_API_KEY'],
            **kwargs
        )
        return resp["output"]["results"]

