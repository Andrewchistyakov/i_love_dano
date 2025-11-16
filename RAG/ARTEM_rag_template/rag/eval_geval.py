import json
from pathlib import Path
from statistics import mean
from typing import Dict, Any, List

from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    GEval,
)
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

from .config import AppConfig
from .pipeline import RAGPipeline


def run_geval_eval(cfg: AppConfig, pipeline: RAGPipeline, dataset_path: str) -> None:
    dataset_path = Path(dataset_path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Eval dataset not found: {dataset_path}")

    results_dir = Path(cfg.eval.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / "geval_results.jsonl"

    eval_model = cfg.eval.eval_model or cfg.llm.model

    # Метрики
    answer_rel = AnswerRelevancyMetric(model=eval_model)
    faithfulness = FaithfulnessMetric(model=eval_model)
    correctness = GEval(
        name="Correctness",
        model=eval_model,
        evaluation_params=[
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.EXPECTED_OUTPUT,
        ],
        evaluation_steps=[
            "Проверь, противоречат ли факты в 'actual output' фактам в 'expected output'.",
            "Сильно штрафуй пропуски ключевых фактов.",
            "Лёгкая разница в формулировках допустима.",
        ],
        threshold=0.6,
    )

    print(f"📊 Запускаем eval по датасету {dataset_path} с моделью {eval_model}")
    scores_rel: List[float] = []
    scores_faith: List[float] = []
    scores_corr: List[float] = []

    with open(dataset_path, "r", encoding="utf-8") as f_in, \
         open(out_path, "w", encoding="utf-8") as f_out:

        for line in f_in:
            if not line.strip():
                continue
            item = json.loads(line)

            q = item["query"]
            expected = item.get("expected_answer", "")

            rag_result = pipeline.answer(q)
            answer = rag_result["answer"]
            contexts = [c["text"] for c in rag_result["contexts"]]

            # LLMTestCase для RAG
            tc = LLMTestCase(
                input=q,
                actual_output=answer,
                expected_output=expected or None,
                retrieval_context=contexts,
                context=item.get("gold_context", None),
            )

            # считаем метрики
            answer_rel.measure(tc)
            faithfulness.measure(tc)
            correctness.measure(tc)

            res_record: Dict[str, Any] = {
                "id": item.get("id"),
                "query": q,
                "expected_answer": expected,
                "answer": answer,
                "contexts": contexts,
                "metrics": {
                    "answer_relevancy": {
                        "score": answer_rel.score,
                        "reason": getattr(answer_rel, "reason", None),
                    },
                    "faithfulness": {
                        "score": faithfulness.score,
                        "reason": getattr(faithfulness, "reason", None),
                    },
                    "correctness": {
                        "score": correctness.score,
                        "reason": getattr(correctness, "reason", None),
                    },
                },
            }
            f_out.write(json.dumps(res_record, ensure_ascii=False) + "\n")

            scores_rel.append(answer_rel.score)
            scores_faith.append(faithfulness.score)
            scores_corr.append(correctness.score)

    print(f"✅ Результаты eval сохранены в {out_path}")
    if scores_rel:
        print(f"AnswerRelevancy: mean={mean(scores_rel):.3f}")
        print(f"Faithfulness:    mean={mean(scores_faith):.3f}")
        print(f"Correctness:     mean={mean(scores_corr):.3f}")