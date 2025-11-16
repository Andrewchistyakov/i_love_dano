from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Dict, Optional
import torch

from sentence_transformers import CrossEncoder
from FlagEmbedding import FlagReranker  # pip install FlagEmbedding

from .config import AppConfig, RerankerConfig


class BaseReranker(ABC):
    @abstractmethod
    def rerank(self, query: str, contexts: List[Dict]) -> List[Dict]:
        pass


class CrossEncoderReranker(BaseReranker):
    """
    Старый одномодельный реранкер: просто пересортировывает top_k, не меняя размер списка.
    Оставлен как fallback.
    """

    def __init__(self, cfg: RerankerConfig):
        self.cfg = cfg
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = CrossEncoder(
            cfg.model_name, 
            device=self.device,
            trust_remote_code=True
        )

    def rerank(self, query: str, contexts: List[Dict]) -> List[Dict]:
        if not contexts or len(contexts) == 1:
            return contexts

        # ограничиваем число кандидатов
        top_k_in = min(self.cfg.top_k, len(contexts))
        candidates = contexts[:top_k_in]

        pairs = [(query, c["text"]) for c in candidates]
        scores = self.model.predict(pairs)

        for c, s in zip(candidates, scores):
            c["rerank_score"] = float(s)

        candidates_sorted = sorted(candidates, key=lambda x: -x["rerank_score"])

        # если исходный список был длиннее — доклеиваем хвост без изменений
        tail = contexts[top_k_in:]
        return candidates_sorted + tail


class TwoStageReranker(BaseReranker):
    """
    Двухступенчатый реранкер, как в ноутбуке:

    Вход: до 500 кандидатов от retriever'а.
      1-й этап: CrossEncoder (Jina) → top first_top_k (обычно 200)
      2-й этап: CrossEncoder или BGE FlagReranker → top final_top_k (обычно 8)

    На выходе: ровно final_top_k контекстов.
    """

    def __init__(self, cfg: RerankerConfig):
        self.cfg = cfg
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        if not cfg.first_model_name or not cfg.second_model_name:
            raise ValueError(
                "TwoStageReranker требует first_model_name и second_model_name в RerankerConfig"
            )

        # 1-й реранкер (обычно Jina)
        self.first = CrossEncoder(
            cfg.first_model_name,
            device=self.device,
            trust_remote_code=True
        )

        # 2-й реранкер (BGE или ещё один CrossEncoder)
        if "bge-reranker" in cfg.second_model_name:
            self.second_type = "flag"
            self.second = FlagReranker(cfg.second_model_name, use_fp16=False)
        else:
            self.second_type = "crossencoder"
            self.second = CrossEncoder(cfg.second_model_name)

    def rerank(self, query: str, contexts: List[Dict]) -> List[Dict]:
        if not contexts:
            return contexts

        # ---------- 1-й этап ----------
        pairs = [(query, c["text"]) for c in contexts]
        scores1 = self.first.predict(pairs)

        # прикрепляем score и сортируем
        for c, s in zip(contexts, scores1):
            c["rerank_score_stage1"] = float(s)

        contexts_sorted = sorted(contexts, key=lambda x: -x["rerank_score_stage1"])

        first_top_k = min(self.cfg.first_top_k, len(contexts_sorted))
        stage1 = contexts_sorted[:first_top_k]

        # ---------- 2-й этап ----------
        pairs2 = [(query, c["text"]) for c in stage1]

        if self.second_type == "crossencoder":
            scores2 = self.second.predict(pairs2)
        else:
            scores2 = self.second.compute_score(pairs2)

        for c, s in zip(stage1, scores2):
            c["rerank_score_stage2"] = float(s)

        stage1_sorted = sorted(stage1, key=lambda x: -x["rerank_score_stage2"])

        final_top_k = min(self.cfg.final_top_k, len(stage1_sorted))
        final_contexts = stage1_sorted[:final_top_k]

        return final_contexts


def create_reranker(cfg: AppConfig) -> Optional[BaseReranker]:
    rcfg = cfg.reranker
    if not rcfg.enabled:
        return None

    # если заданы две модели — используем двухступенчатый реранкер
    if rcfg.first_model_name and rcfg.second_model_name:
        return TwoStageReranker(rcfg)

    # иначе — старый одномодельный
    return CrossEncoderReranker(rcfg)