# rag/experiments.py
from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import yaml
import numpy as np

from .config import AppConfig, load_config
from .index import load_index
from .retrievers import create_retriever
from .llms import create_llm
from .pipeline import RAGPipeline
from .reranker import create_reranker
from .ranker import create_ranker
from .eval_geval import run_geval_eval


# ---------- Вспомогалки ----------

def apply_overrides(cfg: AppConfig, overrides: Dict[str, Any]) -> AppConfig:
    """
    Применяет overrides вида:
      {
        "retriever.type": "hybrid",
        "ranker.enabled": true,
        "ranker.mmr_lambda": 0.5,
        "index.top_k": 15
      }
    к dataclass-конфигу.
    """
    cfg = copy.deepcopy(cfg)

    for key, value in overrides.items():
        parts = key.split(".")
        obj = cfg
        for p in parts[:-1]:
            if not hasattr(obj, p):
                raise AttributeError(f"Config has no field '{p}' in path '{key}'")
            obj = getattr(obj, p)
        last = parts[-1]
        if not hasattr(obj, last):
            raise AttributeError(f"Config has no field '{last}' in path '{key}'")
        setattr(obj, last, value)
    return cfg


def build_pipeline(cfg: AppConfig) -> RAGPipeline:
    index = load_index(cfg)
    retriever = create_retriever(cfg, index)
    llm = create_llm(cfg.llm)
    ranker = create_ranker(cfg)
    reranker = None
    if ranker is None:
        reranker = create_reranker(cfg)
    return RAGPipeline(cfg, retriever, llm, reranker=reranker, ranker=ranker)


def summarize_eval_results(results_path: Path) -> Dict[str, float]:
    """
    Читает eval_results/*.jsonl и считает средние метрики.
    Формат строк — как в eval_geval.py.
    """
    if not results_path.exists():
        return {}
    scores_rel, scores_faith, scores_corr = [], [], []
    with open(results_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            m = item.get("metrics", {})
            if "answer_relevancy" in m:
                scores_rel.append(m["answer_relevancy"]["score"])
            if "faithfulness" in m:
                scores_faith.append(m["faithfulness"]["score"])
            if "correctness" in m:
                scores_corr.append(m["correctness"]["score"])
    out = {}
    if scores_rel:
        out["answer_relevancy"] = float(np.mean(scores_rel))
    if scores_faith:
        out["faithfulness"] = float(np.mean(scores_faith))
    if scores_corr:
        out["correctness"] = float(np.mean(scores_corr))
    return out


# ---------- Структура эксперимента ----------

@dataclass
class Experiment:
    name: str
    overrides: Dict[str, Any]


def load_experiments_yaml(path: str) -> List[Experiment]:
    """
    Ожидаемый формат experiments.yaml:

    base_results_dir: eval_results
    experiments:
      - name: vec_ce
        overrides:
          retriever.type: vector
          reranker.enabled: true
          ranker.enabled: false
          index.top_k: 20
      - name: hybrid_ranker_mmr
        overrides:
          retriever.type: hybrid
          ranker.enabled: true
          reranker.enabled: false
          ranker.mmr_lambda: 0.5
          index.top_k: 30
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)
    with open(p, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    exps_raw = data.get("experiments", [])
    if not exps_raw:
        raise ValueError("experiments.yaml: нет секции 'experiments'")

    experiments: List[Experiment] = []
    for e in exps_raw:
        experiments.append(
            Experiment(
                name=e["name"],
                overrides=e.get("overrides", {}),
            )
        )

    base_results_dir = data.get("base_results_dir")
    return experiments, base_results_dir


def run_single_experiment(
    base_cfg: AppConfig,
    exp: Experiment,
    dataset_path: str | None,
    base_results_dir: str | None,
) -> Dict[str, Any]:
    """
    Запускает один эксперимент:
      - применяет overrides
      - выставляет отдельную папку results_dir
      - гоняет eval
      - возвращает summary по метрикам
    """
    cfg = apply_overrides(base_cfg, exp.overrides)

    # Настраиваем отдельную папку для результатов
    if base_results_dir:
        from pathlib import Path
        cfg.eval.results_dir = str(Path(base_results_dir) / exp.name)

    ds_path = dataset_path or cfg.eval.dataset_path

    print(f"\n==================== Experiment: {exp.name} ====================")
    print("Overrides:")
    for k, v in exp.overrides.items():
        print(f"  {k} = {v}")

    pipeline = build_pipeline(cfg)
    run_geval_eval(cfg, pipeline, ds_path)

    # считаем summary из сохранённого файла
    results_path = Path(cfg.eval.results_dir) / "geval_results.jsonl"
    metrics = summarize_eval_results(results_path)

    print(f"📊 Summary for {exp.name}: {metrics}")
    return {
        "name": exp.name,
        "metrics": metrics,
        "results_path": str(results_path),
    }


def run_experiments(
    config_path: str = "config.yaml",
    experiments_yaml: str = "experiments.yaml",
    dataset_path: str | None = None,
) -> List[Dict[str, Any]]:
    base_cfg = load_config(config_path)
    experiments, base_results_dir = load_experiments_yaml(experiments_yaml)

    all_results: List[Dict[str, Any]] = []
    for exp in experiments:
        res = run_single_experiment(base_cfg, exp, dataset_path, base_results_dir)
        all_results.append(res)

    # итоговая табличка
    print("\n=========== EXPERIMENTS SUMMARY ===========")
    for r in all_results:
        name = r["name"]
        m = r["metrics"]
        print(f"{name:25s} | ", end="")
        print(
            " ".join(
                f"{k}={v:.3f}" for k, v in m.items()
            )
        )

    return all_results