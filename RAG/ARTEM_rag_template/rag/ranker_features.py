from typing import List, Dict, Tuple
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from .config import AppConfig
from .embeddings import embed_texts
from .index import _tokenize


def build_features_for_query(
    cfg: AppConfig,
    query: str,
    candidates: List[Dict],
    ce_model: CrossEncoder | None = None,
) -> Tuple[np.ndarray, list[str], np.ndarray]:
    """
    Строит признаки для (query, candidates):
      - sim_dense (cosine)
      - bm25_score (по корпусу candidate-ов)
      - bm25_norm
      - hybrid_score (alpha * dense + (1-alpha)*bm25_norm)
      - len_chars, len_tokens
      - token_overlap_count, token_overlap_ratio, jaccard
      - retrieval_rank
      - ce_score (если ce_model не None)

    Возвращает: (X, feature_names, chunk_embeddings)
    """
    if not candidates:
        return np.zeros((0, 0)), [], np.zeros((0, 0))

    texts = [c["text"] for c in candidates]
    tokens_q = _tokenize(query)
    tokens_list = [_tokenize(t) for t in texts]

    # BM25 по кандидату-мультисету
    bm25 = BM25Okapi(tokens_list)
    bm25_scores = bm25.get_scores(tokens_q)
    bmin, bmax = float(np.min(bm25_scores)), float(np.max(bm25_scores))
    bden = (bmax - bmin) + 1e-8
    bm25_norm = (bm25_scores - bmin) / bden

    # dense sim: эмбеддинг query + чанков
    embs = embed_texts(cfg.embedding.model_name, [query] + texts)
    q_emb = embs[0]
    chunk_embs = embs[1:]
    dense_scores = chunk_embs @ q_emb

    alpha = cfg.retriever.hybrid_alpha
    hybrid_scores = alpha * dense_scores + (1 - alpha) * bm25_norm

    # cross-encoder score
    if ce_model is not None:
        pairs = [(query, t) for t in texts]
        ce_scores = ce_model.predict(pairs)
    else:
        ce_scores = np.zeros(len(texts), dtype="float32")

    feature_names = [
        "sim_dense",
        "bm25_score",
        "bm25_norm",
        "hybrid_score",
        "len_chars",
        "len_tokens",
        "overlap_count",
        "overlap_ratio",
        "jaccard",
        "retrieval_rank",
        "ce_score",
    ]

    X = []
    for i, (toks, text) in enumerate(zip(tokens_list, texts)):
        len_chars = len(text)
        len_tokens = len(toks)

        set_q = set(tokens_q)
        set_c = set(toks)
        inter = set_q & set_c
        union = set_q | set_c
        overlap_count = len(inter)
        overlap_ratio = overlap_count / (len(set_q) + 1e-8)
        jaccard = overlap_count / (len(union) + 1e-8)

        retrieval_rank = candidates[i].get("retrieval_rank", i)

        row = [
            float(dense_scores[i]),
            float(bm25_scores[i]),
            float(bm25_norm[i]),
            float(hybrid_scores[i]),
            float(len_chars),
            float(len_tokens),
            float(overlap_count),
            float(overlap_ratio),
            float(jaccard),
            float(retrieval_rank),
            float(ce_scores[i]),
        ]
        X.append(row)

    X = np.asarray(X, dtype="float32")
    return X, feature_names, chunk_embs