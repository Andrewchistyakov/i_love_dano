# make_submission.py
#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml

from rag.config import load_config, AppConfig
from rag.experiments import apply_overrides
from rag.index import load_index
from rag.retrievers import create_retriever
from rag.llms import create_llm
from rag.reranker import create_reranker
from rag.ranker import create_ranker
from rag.pipeline import RAGPipeline


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


def build_pipeline(cfg: AppConfig) -> RAGPipeline:
    index = load_index(cfg)
    retriever = create_retriever(cfg, index)
    llm = create_llm(cfg.llm)
    ranker = create_ranker(cfg)
    reranker = None
    if ranker is None:
        reranker = create_reranker(cfg)
    return RAGPipeline(cfg, retriever, llm, reranker=reranker, ranker=ranker)


# ====== Место, где мы достаём sections/pages из чанков ======

def extract_section_and_page_from_source(source: str) -> Tuple[str | None, str | None]:
    """
    Хелпер, который по пути к исходному файлу (c['source']) пытается восстановить:
      - section: строка вроде "psychological_research/approaches_to_research"
      - page: строка с номером страницы, например "41"

    ⚠️ Очень важно:
    - Здесь нужно подстроить логику под твою реальную структуру файлов.
    - Сейчас стоит разумный дефолт: считаем, что путь примерно
        data/docs/<section>/<page>.<ext>
      и берём:
        section = "<section>" или "subdir/subsubdir"
        page = "<stem файла>" (с отрезанием префикса 'page_' если есть)
    """
    p = Path(source)

    # Попробуем найти кусок пути после "data/docs"
    parts = p.parts
    section = None
    page = None

    try:
        # ищем "data" и "docs" в пути
        if "data" in parts:
            idx_data = parts.index("data")
            # если дальше есть "docs" — ищем после него
            if "docs" in parts[idx_data + 1:]:
                idx_docs = parts.index("docs", idx_data + 1)
                # всё, что после docs, кроме последнего элемента (файла) — считаем section-путём
                section_parts = parts[idx_docs + 1:-1]
                if section_parts:
                    section = "/".join(section_parts)
        # fallback: просто родительская папка
        if section is None:
            section = p.parent.name

        # page: из имени файла
        stem = p.stem  # например "41" или "page_41"
        if stem.lower().startswith("page_"):
            stem = stem[5:]
        page = stem
    except Exception:
        # Если что-то пошло не так — лучше вернуть хоть что-то
        section = p.parent.name
        page = p.stem

    return section, page


def build_references_from_contexts(contexts: List[Dict]) -> Dict[str, List[str]]:
    """
    Строим структуру:
      {
        "sections": [...],
        "pages": [...]
      }
    по списку чанков, которые вернул RAG-пайплайн.
    """
    sections: List[str] = []
    pages: List[str] = []

    for c in contexts:
        src = c.get("source", "")
        section, page = extract_section_and_page_from_source(src)
        if section and section not in sections:
            sections.append(section)
        if page and page not in pages:
            pages.append(page)

    return {
        "sections": sections,
        "pages": pages,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Generate competition submission file (ID,context,answer,references) from RAG pipeline."
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
        help="JSONL с полями {ID, query} (обрати внимание: поле 'ID', а не 'id')",
    )
    parser.add_argument(
        "--out-path",
        type=str,
        default="submission.csv",
        help="Путь до файла сабмита (CSV)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Override на количество контекстов (если нужно изменить config.index.top_k)",
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

    cfg = load_config(args.config)

    # Применяем overrides от выбранного эксперимента (лучшего режима)
    if args.experiment_name is not None:
        overrides = load_experiment_overrides(args.experiments_yaml, args.experiment_name)
        cfg = apply_overrides(cfg, overrides)

    pipeline = build_pipeline(cfg)

    test_path = Path(args.test_path)
    if not test_path.exists():
        raise FileNotFoundError(test_path)

    print(f"📥 Читаем тест из {test_path}")
    if test_path.suffix == ".jsonl":
        rows = []
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
                        "query": item["query"],
                    }
                )
    elif test_path.suffix == ".json":
        rows = []
        with open(test_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            for item in data:
                qid = item.get("ID") or item.get("id") or item.get("query_id")
                if qid is None:
                    raise ValueError(f"Строка теста без ID: {item}")
                rows.append(
                    {
                        "ID": qid,
                        "query": item["question"],
                    }
                )
    else:
        raise ValueError(f"Неизвестный формат файла: {test_path}")

    print(f"📦 Найдено {len(rows)} тестовых запросов")

    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"📤 Пишем сабмит в {out_path}")

    with open(out_path, "w", encoding="utf-8", newline="") as f_out:
        writer = csv.writer(f_out)
        # Строго по формату задачи:
        # ID,context,answer,references
        writer.writerow(["ID", "context", "answer", "references"]) #оставить только то что надо

        for i, item in enumerate(rows, 1):
            qid = item["ID"]
            query = item["query"]

            result = pipeline.answer(query, top_k=args.top_k)
            contexts = result["contexts"]
            answer = result["answer"]
            
            # context — это склейка текста всех выбранных чанков, закомментить при не надобности
            context_text = "\n\n".join(c["text"] for c in contexts)

            # references — JSON: {"sections": [...], "pages": [...]}
            refs = build_references_from_contexts(contexts)
            refs_str = json.dumps(refs, ensure_ascii=False)

            writer.writerow([qid, context_text, answer, refs_str]) #убрать что не нужно

            if i % 20 == 0 or i == len(rows):
                print(f"  ✓ обработано {i}/{len(rows)}")

    print("✅ Готово. submission.csv в нужном формате.")
    

if __name__ == "__main__":
    main()