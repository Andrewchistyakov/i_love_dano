#!/usr/bin/env python3
import argparse

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

from rag.config import load_config
from rag.index import build_index, load_index
from rag.retrievers import create_retriever
from rag.llms import create_llm
from rag.pipeline import RAGPipeline
from rag.eval_geval import run_geval_eval
from rag.reranker import create_reranker
from rag.ranker import create_ranker


def make_pipeline(cfg):
    index = load_index(cfg)
    retriever = create_retriever(cfg, index)
    llm = create_llm(cfg.llm)
    ranker = create_ranker(cfg)
    reranker = None
    if ranker is None:
        reranker = create_reranker(cfg)
    return RAGPipeline(cfg, retriever, llm, reranker=reranker, ranker=ranker)


def main():
    parser = argparse.ArgumentParser(
        description="RAG with semantic chunking, FAISS, supervised ranker, MMR and DeepEval."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_build = subparsers.add_parser("build-index", help="Собрать индекс по документам")
    p_build.add_argument(
        "--docs-path",
        type=str,
        default="data/docs",
        help="Папка с исходными документами (.txt/.md)",
    )

    p_query = subparsers.add_parser("query", help="Задать вопрос RAG-пайплайну")
    p_query.add_argument("question", type=str, help="Вопрос к базе знаний")
    p_query.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Сколько чанков контекста вернуть (override config.index.top_k)",
    )

    p_eval = subparsers.add_parser("eval", help="Запустить оценку через G-Eval/DeepEval")
    p_eval.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="JSONL с тестовыми кейсами; по умолчанию config.eval.dataset_path",
    )

    args = parser.parse_args()
    cfg = load_config("config.yaml")

    if args.command == "build-index":
        build_index(cfg, docs_path=args.docs_path)

    elif args.command == "query":
        pipeline = make_pipeline(cfg)
        result = pipeline.answer(args.question, top_k=args.top_k)

        print("\n=== ОТВЕТ ===\n")
        print(result["answer"])
        print("\n=== ИСПОЛЬЗОВАННЫЙ КОНТЕКСТ ===")
        for i, ctx in enumerate(result["contexts"], 1):
            base_score = ctx.get("score")
            ranker_score = ctx.get("ranker_score")
            rerank_score = ctx.get("rerank_score")
            parts = []
            if base_score is not None:
                parts.append(f"retriever={base_score:.4f}")
            if ranker_score is not None:
                parts.append(f"ranker={ranker_score:.4f}")
            if rerank_score is not None:
                parts.append(f"ce={rerank_score:.4f}")
            score_str = ", ".join(parts)
            print(
                f"\n[{i}] source={ctx['source']} chunk_id={ctx['chunk_id']} "
                f"retrieval_rank={ctx.get('retrieval_rank')} {score_str}"
            )
            print(ctx["text"])

    elif args.command == "eval":
        dataset_path = args.dataset or cfg.eval.dataset_path
        pipeline = make_pipeline(cfg)
        run_geval_eval(cfg, pipeline, dataset_path)


if __name__ == "__main__":
    main()