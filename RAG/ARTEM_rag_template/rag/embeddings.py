from typing import List
from functools import lru_cache
import os

import numpy as np
from sentence_transformers import SentenceTransformer
import torch

# Чтоб не плодить кучу потоков в токенайзере
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


@lru_cache(maxsize=2)
def get_embedding_model(model_name: str) -> SentenceTransformer:
    # Ограничиваем количество потоков BLAS/PyTorch — это уменьшает шанс segfault'ов
    torch.set_num_threads(1)
    return SentenceTransformer(model_name)


def embed_texts(model_name: str, texts: List[str]) -> np.ndarray:
    model = get_embedding_model(model_name)

    # Никаких num_workers — твоя версия encode их не поддерживает
    emb = model.encode(
        texts,
        batch_size=16,              # можешь увеличить, если ОЗУ позволяет
        convert_to_numpy=True,
        show_progress_bar=True,
    )

    # Нормируем, чтобы можно было использовать dot как cosine similarity
    norms = np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8
    emb = emb / norms
    return emb.astype("float32")