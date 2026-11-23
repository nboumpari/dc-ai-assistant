"""Local embedding models and rerankers for the rules assistant.

All models run on the CPU through PyTorch inside the Docker `app` service.

Embedders (vectors are L2-normalised, so dot product = cosine similarity):
  gemma   google/embeddinggemma-300m                 308M parameters, 768 dims, query/document prompts
  granite ibm-granite/granite-embedding-english-r2   149M parameters, 768 dims, no prompts

Rerankers (cross-encoders: score one question-passage pair at a time, higher = more relevant):
  gte     Alibaba-NLP/gte-reranker-modernbert-base    149M parameters, reads whole passages
  minilm  cross-encoder/ms-marco-MiniLM-L6-v2          22M parameters, reads the first 512 tokens

Models are loaded once and cached, in float32 (bfloat16 weights are much slower
on CPUs without bfloat16 support, and EmbeddingGemma does not support float16).
"""

from dataclasses import dataclass
from functools import cache

import numpy as np

from dc_ai_assistant import settings  # also loads .env, e.g. HF_TOKEN for the gated model

MAX_TOKENS = 2048  # chunks are at most ~4,000 characters (~1,000 tokens), so nothing is cut


@dataclass(frozen=True)
class EmbedderSpec:
    model: str
    dims: int
    query_prompt_name: str | None = None  # the model's own prompt for questions, if it has one
    document_prompt_name: str | None = None  # the model's own prompt for passages, if it has one


@dataclass(frozen=True)
class RerankerSpec:
    model: str
    max_tokens: int  # question and passage together; longer passages are cut at the end


EMBEDDERS = {
    "gemma": EmbedderSpec("google/embeddinggemma-300m", 768, "query", "document"),
    "granite": EmbedderSpec("ibm-granite/granite-embedding-english-r2", 768),
}

RERANKERS = {
    "gte": RerankerSpec("Alibaba-NLP/gte-reranker-modernbert-base", MAX_TOKENS),
    "minilm": RerankerSpec("cross-encoder/ms-marco-MiniLM-L6-v2", 512),  # the model's own limit
}


def _float32() -> dict:
    import torch

    torch.set_num_threads(settings.TORCH_THREADS)
    return {"dtype": torch.float32}


@cache
def _sentence_transformer(name: str):
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(name, device="cpu", model_kwargs=_float32())
    model.max_seq_length = MAX_TOKENS
    return model


def embed_documents(embedder: str, texts: list[str], batch_size: int = 16) -> np.ndarray:
    spec = EMBEDDERS[embedder]
    model = _sentence_transformer(spec.model)
    kwargs = {"prompt_name": spec.document_prompt_name} if spec.document_prompt_name else {}
    return model.encode(texts, batch_size=batch_size, normalize_embeddings=True, convert_to_numpy=True, **kwargs)


def embed_query(embedder: str, question: str) -> np.ndarray:
    spec = EMBEDDERS[embedder]
    model = _sentence_transformer(spec.model)
    kwargs = {"prompt_name": spec.query_prompt_name} if spec.query_prompt_name else {}
    return model.encode([question], normalize_embeddings=True, convert_to_numpy=True, **kwargs)[0]


@cache
def _cross_encoder(reranker: str):
    from sentence_transformers import CrossEncoder

    spec = RERANKERS[reranker]
    return CrossEncoder(spec.model, device="cpu", max_length=spec.max_tokens, model_kwargs=_float32())


def rerank_scores(reranker: str, question: str, passages: list[str]) -> list[float]:
    model = _cross_encoder(reranker)
    return [float(s) for s in model.predict([(question, p) for p in passages], batch_size=8)]
