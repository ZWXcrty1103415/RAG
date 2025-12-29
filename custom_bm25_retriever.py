
"""
The BM25Retriever class from langchain does not provide a function that allow external tokenized text as its input.
Using a pre-tokenized text list built in the data_process session can largely improve the loading time of the system.
"""

from langchain_community.retrievers import BM25Retriever
from typing import Any, Callable, Dict, Iterable, List, Optional
from langchain_core.documents import Document

def default_preprocessing_func(text: str) -> List[str]:
    return text.split()

class BM25Retriever2(BM25Retriever):
    @classmethod
    def from_tokenized_text(
            cls,
            texts: Iterable[str],
            tokenized_texts: Iterable[List[str]],
            metadatas: Optional[Iterable[dict]] = None,
            ids: Optional[Iterable[str]] = None,
            bm25_params: Optional[Dict[str, Any]] = None,
            preprocess_func: Callable[[str], List[str]] = default_preprocessing_func,
            **kwargs: Any,
    ) -> BM25Retriever:
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            raise ImportError(
                "Could not import rank_bm25, please install with `pip install "
                "rank_bm25`."
            )

        texts_processed = tokenized_texts
        bm25_params = bm25_params or {}
        vectorizer = BM25Okapi(texts_processed, **bm25_params)
        metadatas = metadatas or ({} for _ in texts)
        if ids:
            docs = [
                Document(page_content=t, metadata=m, id=i)
                for t, m, i in zip(texts, metadatas, ids)
            ]
        else:
            docs = [
                Document(page_content=t, metadata=m) for t, m in zip(texts, metadatas)
            ]
        return cls(
            vectorizer=vectorizer, docs=docs, preprocess_func=preprocess_func, **kwargs
        )
