from typing import Dict, List, Optional

from .config import AppConfig
from .retrievers import BaseRetriever
from .llms import BaseLLMClient
from .reranker import BaseReranker
from .ranker import BaseRanker


class RAGPipeline:
    def __init__(
        self,
        cfg: AppConfig,
        retriever: BaseRetriever,
        llm: BaseLLMClient,
        reranker: Optional[BaseReranker] = None,
        ranker: Optional[BaseRanker] = None,
    ):
        self.cfg = cfg
        self.retriever = retriever
        self.llm = llm
        self.reranker = reranker
        self.ranker = ranker

    @staticmethod
    def build_prompt(question: str, contexts: List[Dict]) -> str:
        context_text = "\n\n".join(
            f"[{i+1}] {c['text']}" for i, c in enumerate(contexts)
        )
        prompt = (
            "Ты — ассистент, отвечающий на вопросы по базе знаний.\n"
            "Используй только факты из контекста. "
            "Если в контексте нет ответа, честно скажи, что не знаешь.\n\n"
            f"Контекст:\n{context_text}\n\n"
            f"Вопрос: {question}\n\n"
            "Дай короткий, точный ответ на русском языке:"
        )
        return prompt

    def answer(self, question: str, top_k: Optional[int] = None) -> Dict:
        k = top_k or self.cfg.index.top_k
        print("Top k:", k)

        # 1) базовый retriever
        contexts = self.retriever.retrieve(question, self.cfg.retriever.n_candidates)
        print("after retriever:", len(contexts))

        # 2) supervised ranker, если включен
        if self.ranker is not None:
            contexts = self.ranker.rank_and_select(question, contexts)
            print("after ranker:", len(contexts))

        # 3) иначе — обычный cross-encoder rerank (старый вариант)
        elif self.reranker is not None:
            contexts = self.reranker.rerank(question, contexts)
        print(f"🔍 Retrieved {len(contexts)} contexts after rerank")

        prompt = self.build_prompt(question, contexts)
        answer = self.llm.generate(prompt)
        return {
            "answer": answer,
            "contexts": contexts,
            "prompt": prompt,
        }