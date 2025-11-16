import os
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path

import yaml

from dataclasses import dataclass
from typing import Optional


@dataclass
class ChunkingConfig:
    type: str = "token"           # "token" | "semantic" ...
    max_tokens: int = 450
    overlap_tokens: int = 90
    tokenizer_model: str = "intfloat/multilingual-e5-large"
    # старые поля (если нужны) тоже можно оставить:
    max_chars: int = 800
    overlap_ratio: float = 0.3
    sentence_min_chars: int = 40

@dataclass
class RetrieverConfig:
    type: str = "hybrid"          # уже было
    hybrid_alpha: float = 0.5
    n_candidates: int = 500       # количество кандидатов до реранка

@dataclass
class RerankerConfig:
    enabled: bool = True
    first_model_name: str = "jinaai/jina-reranker-v2-base-multilingual"
    first_top_k: int = 200
    second_model_name: str = "BAAI/bge-reranker-v2-m3"
    final_top_k: int = 8          # финальное число контекстов в LLM

@dataclass
class EmbeddingConfig:
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"

@dataclass
class IndexConfig:
    index_dir: str = "storage/index"
    top_k: int = 5
    vector_backend: str = "numpy"
    faiss_hnsw_m: int = 32


@dataclass
class LLMConfig:
    provider: str = "openai"  # openai | tgi | huggingface | local
    model: str = "gpt-4o-mini"
    temperature: float = 0.2
    max_tokens: int = 512
    base_url: Optional[str] = None
    api_key_env: str = "OPENAI_API_KEY"
    device: Optional[str] = None      # "cuda", "mps", "cpu" — для local
    dtype: Optional[str] = None   


@dataclass
class RankerConfig:
    enabled: bool = False
    model_type: str = "lgbm"  # lgbm | catboost
    model_path: str = "storage/ranker/ranker_model.pkl"
    use_cross_encoder_feature: bool = True
    selection_mode: str = "fixed_k"  # fixed_k | threshold | dynamic_k
    score_threshold: float = 0.0
    dynamic_k_model_path: str = "storage/ranker/dynamic_k_model.pkl"
    mmr_lambda: float = 0.5
    mmr_top_k_candidates: int = 20


@dataclass
class EvalConfig:
    dataset_path: str = "data/eval/rag_eval_example.jsonl"
    results_dir: str = "eval_results"
    eval_model: Optional[str] = None


@dataclass
class AppConfig:
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    index: IndexConfig = field(default_factory=IndexConfig)
    retriever: RetrieverConfig = field(default_factory=RetrieverConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    reranker: RerankerConfig = field(default_factory=RerankerConfig)
    ranker: RankerConfig = field(default_factory=RankerConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)


def load_config(path: str = "config.yaml") -> AppConfig:
    cfg = AppConfig()
    if not Path(path).exists():
        return cfg

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    if "embedding" in data:
        for k, v in data["embedding"].items():
            setattr(cfg.embedding, k, v)

    if "chunking" in data:
        for k, v in data["chunking"].items():
            setattr(cfg.chunking, k, v)

    if "index" in data:
        for k, v in data["index"].items():
            setattr(cfg.index, k, v)

    if "retriever" in data:
        for k, v in data["retriever"].items():
            setattr(cfg.retriever, k, v)

    if "llm" in data:
        for k, v in data["llm"].items():
            setattr(cfg.llm, k, v)

    if "reranker" in data:
        for k, v in data["reranker"].items():
            setattr(cfg.reranker, k, v)

    if "ranker" in data:
        for k, v in data["ranker"].items():
            setattr(cfg.ranker, k, v)

    if "eval" in data:
        for k, v in data["eval"].items():
            setattr(cfg.eval, k, v)

    cfg.index.index_dir = os.path.expanduser(cfg.index.index_dir)
    cfg.eval.results_dir = os.path.expanduser(cfg.eval.results_dir)

    return cfg