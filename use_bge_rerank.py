
"""
using bge-rerank-large model with huggingface transformers
"""

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

device = "cuda" if torch.cuda.is_available() else "cpu"

class bge_reranker:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-reranker-large", cache_dir="./hf_models", device_map="auto")
        self.model = AutoModelForSequenceClassification.from_pretrained("BAAI/bge-reranker-large", cache_dir="./hf_models", device_map="auto")
        self.model.eval()

    def rerank(self, query:str, documents: list[str], top_n : int = 8, **kwargs):
        pairs = [[query, doc] for doc in documents]
        with torch.no_grad():
            inputs = self.tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=512)
            scores = self.model(**inputs, return_dict=True).logits.view(-1, ).float()
            ranked = scores.argsort()[-top_n:].flip(-1).tolist()
        return ranked, scores.tolist()


if __name__=="__main__":
    tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-reranker-large", cache_dir="./hf_models")
    model = AutoModelForSequenceClassification.from_pretrained("BAAI/bge-reranker-large", cache_dir="./hf_models")
    model.eval()

    pairs = [['what is panda?', 'hi'], ['what is panda?', 'The giant panda (Ailuropoda melanoleuca), sometimes called a panda bear or simply panda, is a bear species endemic to China.']]
    with torch.no_grad():
        inputs = tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=512)
        scores = model(**inputs, return_dict=True).logits.view(-1, ).float()
        print(scores)