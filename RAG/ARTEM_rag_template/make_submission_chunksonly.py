import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

from rag.config import load_config, AppConfig
from rag.experiments import apply_overrides
from rag.index import load_index
from rag.retrievers import create_retriever, BaseRetriever
from rag.reranker import create_reranker, BaseReranker
from rag.ranker import create_ranker, BaseRanker


def load_experiment_overrides(experiments_yaml: str, experiment_name: str) -> Dict[str, Any]:
    p = Path(experiments_yaml)
    if not p.exists():
        raise FileNotFoundError(experiments_yaml)
    with open(p, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    for e in data.get("experiments", []):
        if e["name"] == experiment_name:
            return e.get("overrides", {})
    raise ValueError(f"Experiment '{experiment_name}' not found in {experiments_yaml}")


def build_components(cfg: AppConfig) -> Tuple[BaseRetriever, Optional[BaseReranker], Optional[BaseRanker]]:
    """
    Собираем только то, что нужно для ранжирования чанков:
    - retriever
    - (опционально) supervised ranker
    - (опционально) cross-encoder / two-stage reranker
    """
    index = load_index(cfg)
    retriever = create_retriever(cfg, index)
    ranker = create_ranker(cfg)
    reranker: Optional[BaseReranker] = None
    if ranker is None:
        reranker = create_reranker(cfg)
    return retriever, reranker, ranker


def rank_chunks_for_query(
    cfg: AppConfig,
    retriever: BaseRetriever,
    reranker: Optional[BaseReranker],
    ranker: Optional[BaseRanker],
    query: str,
    top_k: Optional[int] = None,
) -> List[Dict]:
    """
    1) retriever → кандидаты
    2) supervised ranker (если включен)
    3) иначе cross-encoder / two-stage reranker (если включен)
    Возвращает список чанков в порядке убывания приоритета.
    """
    k = top_k or cfg.index.top_k

    # 1) базовый retriever
    contexts = retriever.retrieve(query, k)

    # 2) supervised ranker, если включён
    if ranker is not None:
        contexts = ranker.rank_and_select(query, contexts)

    # 3) иначе — cross-encoder / TwoStageReranker
    elif reranker is not None:
        contexts = reranker.rerank(query, contexts)

    return contexts


def main():
    parser = argparse.ArgumentParser(
        description="Generate submission with ranked chunks: columns [ID, chunk_id, text]."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Base config.yaml",
    )
    parser.add_argument(
        "--test-path",
        type=str,
        default="data/competition/test_queries.jsonl",
        help="JSON/JSONL с полями {ID, query} (обрати внимание: 'ID', а не 'id')",
    )
    parser.add_argument(
        "--out-path",
        type=str,
        default="submission_chunks.csv",
        help="Путь до файла сабмита (CSV)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Сколько чанков брать на запрос (по умолчанию cfg.index.top_k)",
    )
    parser.add_argument(
        "--experiments-yaml",
        type=str,
        default="experiments.yaml",
        help="YAML с экспериментами (для выбора лучшего конфига)",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default=None,
        help="Имя эксперимента из experiments.yaml, чьи overrides применить",
    )
    args = parser.parse_args()

    # ====== 1. Конфиг + overrides ======
    cfg = load_config(args.config)

    if args.experiment_name is not None:
        overrides = load_experiment_overrides(args.experiments_yaml, args.experiment_name)
        cfg = apply_overrides(cfg, overrides)

    # ====== 2. Компоненты (без LLM) ======
    retriever, reranker, ranker = build_components(cfg)

    # ====== 3. Читаем тест ======
    test_path = Path(args.test_path)
    if not test_path.exists():
        raise FileNotFoundError(test_path)

    print(f"📥 Читаем тест из {test_path}")
    rows: List[Dict[str, Any]] = []
    if test_path.suffix == ".jsonl":
        with open(test_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line)
                qid = item.get("ID") or item.get("id") or item.get("query_id")
                if qid is None:
                    raise ValueError(f"Строка теста без ID: {item}")
                rows.append(
                    {
                        "ID": qid,
                        "query": item.get("query") or item.get("question"),
                    }
                )
    elif test_path.suffix == ".json":
        with open(test_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            for item in data:
                qid = item.get("ID") or item.get("id") or item.get("query_id")
                if qid is None:
                    raise ValueError(f"Строка теста без ID: {item}")
                rows.append(
                    {
                        "ID": qid,
                        "query": item.get("query") or item.get("question"),
                    }
                )
    else:
        raise ValueError(f"Неизвестный формат файла: {test_path}")

    print(f"📦 Найдено {len(rows)} тестовых запросов")

    # ====== 4. Сабмит ======
    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"📤 Пишем сабмит в {out_path}")
    with open(out_path, "w", encoding="utf-8", newline="") as f_out:
        writer = csv.writer(f_out)
        # Три колонки:
        #   ID        — ID запроса
        #   chunk_id  — ID чанка (один на строку), строки идут в порядке приоритета
        #   text      — текст чанка
        writer.writerow(["ID", "chunk_id", "text"])

        for i, item in enumerate(rows, 1):
            qid = item["ID"]
            query = item["query"]

            contexts = rank_chunks_for_query(
                cfg,
                retriever=retriever,
                reranker=reranker,
                ranker=ranker,
                query=query,
                top_k=args.top_k,
            )

            # Порядок контекстов = порядок приоритета
            for c in contexts:
                doc_id = c.get("doc_id")
                chunk_id = c.get("chunk_id")

                # формируем стабильный идентификатор чанка
                if doc_id is not None and chunk_id is not None:
                    uid = f"{doc_id}__{chunk_id}"
                else:
                    # fallback на source, если вдруг doc_id нет
                    src = c.get("source", "")
                    uid = f"{src}__{chunk_id}"

                text = c.get("text", "")

                writer.writerow([qid, uid, text])

            if i % 20 == 0 or i == len(rows):
                print(f"  ✓ обработано {i}/{len(rows)}")

    print("✅ Готово. Сабмит с (ID, chunk_id, text) сохранён.")


if __name__ == "__main__":
    main()