from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Dict, Optional

import numpy as np
import joblib
from sentence_transformers import CrossEncoder

from pathlib import Path

from .config import AppConfig
from .ranker_features import build_features_for_query


class BaseRanker(ABC):
    @abstractmethod
    def rank_and_select(self, query: str, candidates: List[Dict]) -> List[Dict]:
        ...


def _mmr_select(
    scores: np.ndarray,
    emb: np.ndarray,
    k: int,
    lambda_mmr: float,
) -> List[int]:
    """
    MMR: iteratively pick item that maximizes
        λ * relevance - (1-λ) * max_sim_to_selected
    """
    n = len(scores)
    if n == 0:
        return []

    k = min(k, n)
    if k <= 0:
        return []

    # косинусное сходство через dot (embeddings уже L2-нормированы)
    sim_matrix = emb @ emb.T

    selected: list[int] = []
    candidate_idx = list(range(n))

    while len(selected) < k and candidate_idx:
        best_i = None
        best_score = -1e9
        for i in candidate_idx:
            relevance = scores[i]
            if not selected:
                diversity_penalty = 0.0
            else:
                diversity_penalty = max(sim_matrix[i, j] for j in selected)
            mmr_score = lambda_mmr * relevance - (1 - lambda_mmr) * diversity_penalty
            if mmr_score > best_score:
                best_score = mmr_score
                best_i = i
        selected.append(best_i)
        candidate_idx.remove(best_i)

    return selected


class SupervisedRanker(BaseRanker):
    """
    Supervised ranker поверх retriever'а:
      - фичи из ranker_features (dense, bm25, hybrid, overlaps, CE и т.д.)
      - LightGBMRanker / CatBoostRanker внутри (модель загружается из ranker.model_path)
      - selection_mode: fixed_k | threshold | dynamic_k
      - MMR поверх предсказанных score'ов
    """

    def __init__(self, cfg: AppConfig):
        self.cfg = cfg
        rcfg = cfg.ranker

        self.model = joblib.load(rcfg.model_path)

        self.dynamic_k_model = None
        if rcfg.selection_mode == "dynamic_k":
            try:
                self.dynamic_k_model = joblib.load(rcfg.dynamic_k_model_path)
            except FileNotFoundError:
                self.dynamic_k_model = None

        self.ce_model: Optional[CrossEncoder] = None
        if rcfg.use_cross_encoder_feature:
            from .config import RerankerConfig  # just for typing
            self.ce_model = CrossEncoder(cfg.reranker.model_name)

    def _select_k(self, scores: np.ndarray) -> int:
        rcfg = self.cfg.ranker
        n = len(scores)
        if n == 0:
            return 0

        if rcfg.selection_mode == "fixed_k":
            return min(self.cfg.index.top_k, n)

        if rcfg.selection_mode == "threshold":
            idx = [i for i, s in enumerate(scores) if s >= rcfg.score_threshold]
            if not idx:
                return min(self.cfg.index.top_k, n)
            return len(idx)

        if rcfg.selection_mode == "dynamic_k" and self.dynamic_k_model is not None:
            max_s = float(np.max(scores))
            mean_s = float(np.mean(scores))
            std_s = float(np.std(scores))
            feat = np.array([[max_s, mean_s, std_s, float(n)]], dtype="float32")
            k_pred = int(round(float(self.dynamic_k_model.predict(feat)[0])))
            k_pred = max(1, min(n, k_pred))
            return k_pred

        # fallback
        return min(self.cfg.index.top_k, n)

    def rank_and_select(self, query: str, candidates: List[Dict]) -> List[Dict]:
        if not candidates:
            return candidates

        # строим фичи
        X, feature_names, chunk_embs = build_features_for_query(
            self.cfg, query, candidates, ce_model=self.ce_model
        )
        scores = self.model.predict(X)
        scores = np.asarray(scores, dtype="float32").reshape(-1)

        # выбираем размер множества
        k = self._select_k(scores)
        if k <= 0:
            k = 1

        # MMR
        rcfg = self.cfg.ranker
        if rcfg.mmr_lambda > 0.0:
            sel_idx = _mmr_select(scores, chunk_embs, k, rcfg.mmr_lambda)
        else:
            sel_idx = np.argsort(-scores)[:k].tolist()

        # собираем результат
        result: List[Dict] = []
        for rank, i in enumerate(sel_idx):
            c = candidates[i].copy()
            c["ranker_score"] = float(scores[i])
            c["ranker_rank"] = rank
            result.append(c)
        return result


def create_ranker(cfg: AppConfig) -> Optional[BaseRanker]:
    """
    Создаём supervised ranker только если:
      - ranker.enabled == True
      - файл с моделью существует

    Если модель не найдена (или ranker.disabled) — возвращаем None,
    и пайплайн автоматически использует cross-encoder или просто retriever.
    """
    if not cfg.ranker.enabled:
        return None

    model_path = Path(cfg.ranker.model_path)
    if not model_path.exists():
        print(
            f"⚠️ ranker.enabled=True, но модель не найдена по пути {model_path}."
            " Supervised ранкер отключён, используем fallback (cross-encoder / retriever)."
        )
        return None

    try:
        return SupervisedRanker(cfg)
    except Exception as e:
        print(
            f"⚠️ Ошибка при загрузке supervised ranker: {e}. "
            "Supervised ранкер отключён, используем fallback."
        )
        return None