from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Dict

from .config import AppConfig
from .embeddings import embed_texts
from .index import SimpleIndex, _tokenize


class BaseRetriever(ABC):
    @abstractmethod
    def retrieve(self, query: str, top_k: int) -> List[Dict]:
        pass


class VectorRetriever(BaseRetriever):
    def __init__(self, cfg: AppConfig, index: SimpleIndex):
        self.cfg = cfg
        self.index = index

    def retrieve(self, query: str, top_k: int) -> List[Dict]:
        if top_k is None:
            top_k = self.cfg.index.top_k
        q_emb = embed_texts(self.cfg.embedding.model_name, [query])[0]
        idxs, scores = self.index.vector_search(q_emb, top_k=top_k)

        results = []
        for rank, (i, score) in enumerate(zip(idxs, scores)):
            meta = self.index.chunks[i].copy()
            meta["score"] = float(score)
            meta["retrieval_rank"] = rank
            results.append(meta)
        return results


class BM25Retriever(BaseRetriever):
    def __init__(self, cfg: AppConfig, index: SimpleIndex):
        self.cfg = cfg
        self.index = index
        if self.index.bm25 is None:
            raise RuntimeError("BM25 retriever requested but bm25 index is missing")

    def retrieve(self, query: str, top_k: int) -> List[Dict]:
        if top_k is None:
            top_k = self.cfg.index.top_k
        tokens = _tokenize(query)
        idxs, scores = self.index.bm25_search(tokens, top_k=top_k)
        results = []
        for rank, (i, score) in enumerate(zip(idxs, scores)):
            meta = self.index.chunks[i].copy()
            meta["score"] = float(score)
            meta["retrieval_rank"] = rank
            results.append(meta)
        return results


class HybridRetriever(BaseRetriever):
    """
    score = alpha * cosine + (1 - alpha) * normalized_bm25
    """

    def __init__(self, cfg: AppConfig, index: SimpleIndex):
        self.cfg = cfg
        self.index = index
        if self.index.bm25 is None:
            raise RuntimeError("Hybrid retriever requested but bm25 index is missing")

    def retrieve(self, query: str, top_k: int) -> List[Dict]:
        if top_k is None:
            top_k = self.cfg.index.top_k

        from collections import defaultdict

        q_emb = embed_texts(self.cfg.embedding.model_name, [query])[0]
        vec_idxs, vec_scores = self.index.vector_search(q_emb, top_k=top_k * 3)

        tokens = _tokenize(query)
        bm25_idxs, bm25_scores = self.index.bm25_search(tokens, top_k=top_k * 3)

        alpha = self.cfg.retriever.hybrid_alpha
        combined = defaultdict(lambda: {"vec": 0.0, "bm25": 0.0})

        if vec_scores:
            vmin, vmax = min(vec_scores), max(vec_scores)
            vden = (vmax - vmin) + 1e-8
            for i, s in zip(vec_idxs, vec_scores):
                combined[i]["vec"] = (s - vmin) / vden

        if bm25_scores:
            bmin, bmax = min(bm25_scores), max(bm25_scores)
            bden = (bmax - bmin) + 1e-8
            for i, s in zip(bm25_idxs, bm25_scores):
                combined[i]["bm25"] = (s - bmin) / bden

        scored = []
        for i, comp in combined.items():
            score = alpha * comp["vec"] + (1 - alpha) * comp["bm25"]
            scored.append((i, score, comp["vec"], comp["bm25"]))

        scored.sort(key=lambda x: -x[1])
        scored = scored[:top_k]

        results: List[Dict] = []
        for rank, (i, score, vec_s, bm25_s) in enumerate(scored):
            meta = self.index.chunks[i].copy()
            meta["score"] = float(score)
            meta["score_dense"] = float(vec_s)
            meta["score_bm25"] = float(bm25_s)
            meta["retrieval_rank"] = rank
            results.append(meta)
        return results


def create_retriever(cfg: AppConfig, index: SimpleIndex) -> BaseRetriever:
    t = cfg.retriever.type.lower()
    if t == "vector":
        return VectorRetriever(cfg, index)
    if t == "bm25":
        return BM25Retriever(cfg, index)
    if t == "hybrid":
        return HybridRetriever(cfg, index)
    raise ValueError(f"Unknown retriever type: {cfg.retriever.type}")