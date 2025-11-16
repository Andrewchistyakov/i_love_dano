import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import numpy as np
from tqdm import tqdm
from rank_bm25 import BM25Okapi

from .config import AppConfig
from .loaders import load_documents
from .chunking import chunk_document
from .embeddings import embed_texts


@dataclass
class SimpleIndex:
    embeddings: np.ndarray
    chunks: List[Dict]
    bm25: Optional[BM25Okapi] = None
    faiss_index: Optional[object] = None  # faiss.Index, но без жёсткого импорта

    def vector_search(self, query_embedding: np.ndarray, top_k: int) -> Tuple[List[int], List[float]]:
        # FAISS backend, если есть
        if self.faiss_index is not None:
            import numpy as np  # локальный import для ясности
            q = np.expand_dims(query_embedding.astype("float32"), axis=0)
            scores, idxs = self.faiss_index.search(q, top_k)
            idxs = idxs[0].tolist()
            scores = scores[0].tolist()
            return idxs, scores

        # простая numpy dot (cosine similarity, так как эмбеддинги нормированы)
        sims = self.embeddings @ query_embedding  # shape: (N,)
        idx = np.argsort(-sims)[:top_k]
        return idx.tolist(), sims[idx].tolist()

    def bm25_search(self, tokens: List[str], top_k: int) -> Tuple[List[int], List[float]]:
        if self.bm25 is None:
            raise RuntimeError("BM25 index is not available")
        scores = self.bm25.get_scores(tokens)
        idx = np.argsort(-scores)[:top_k]
        return idx.tolist(), scores[idx].tolist()


def _tokenize(text: str) -> List[str]:
    return [t for t in text.lower().split() if t.strip()]


def _build_faiss_index(emb: np.ndarray, cfg: AppConfig, index_dir: Path) -> Optional[object]:
    backend = (cfg.index.vector_backend or "numpy").lower()
    if backend == "numpy":
        return None

    try:
        import faiss  # type: ignore
    except ImportError:
        print("⚠️  faiss не установлен, vector_backend будет 'numpy'.")
        return None

    d = emb.shape[1]

    if backend == "faiss_flat":
        index = faiss.IndexFlatIP(d)
    elif backend == "faiss_hnsw":
        m = cfg.index.faiss_hnsw_m
        index = faiss.IndexHNSWFlat(d, m)
        # можно подкрутить efSearch при необходимости
        index.hnsw.efSearch = max(32, cfg.index.top_k * 2)
    else:
        print(f"⚠️  Неизвестный vector_backend={backend}, использую numpy.")
        return None

    index.add(emb.astype("float32"))
    faiss.write_index(index, str(index_dir / "faiss.index"))
    print(f"🧱 FAISS индекс ({backend}) сохранён в {index_dir / 'faiss.index'}")
    return index


def build_index(cfg: AppConfig, docs_path: str = "data/docs") -> None:
    """
    Строит индекс:
      - семантический/символьный чанкинг
      - эмбеддинги
      - FAISS (опционально)
      - BM25
      - сохранение на диск
    """
    index_dir = Path(cfg.index.index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)

    print(f"📚 Загружаем документы из {docs_path} ...")
    docs = load_documents(docs_path)
    all_chunks: List[Dict] = []

    print("✂️  Чанким документы ...")
    for d in tqdm(docs, desc="Docs"):
        chunks = chunk_document(cfg, d["text"], doc_id=d["id"], source=d["path"])
        all_chunks.extend(chunks)

    if not all_chunks:
        raise RuntimeError("Не удалось получить ни одного чанка")

    print(f"📦 Всего чанков: {len(all_chunks)}")

    texts = [c["text"] for c in all_chunks]

    print(f"🧠 Считаем эмбеддинги через {cfg.embedding.model_name} ...")
    emb = embed_texts(cfg.embedding.model_name, texts)

    print("💾 Сохраняем эмбеддинги и метадату индекса ...")
    np.save(index_dir / "embeddings.npy", emb)

    with open(index_dir / "chunks.jsonl", "w", encoding="utf-8") as f:
        for c in all_chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    # FAISS индекс (опционально)
    _build_faiss_index(emb, cfg, index_dir)

    # BM25 индекс
    print("📐 Строим BM25 индекс ...")
    tokenized_corpus = [_tokenize(t) for t in texts]
    bm25 = BM25Okapi(tokenized_corpus)

    # сохраняем bm25
    import pickle

    with open(index_dir / "bm25.pkl", "wb") as f:
        pickle.dump({"bm25": bm25}, f)

    print(f"✅ Индекс готов и сохранён в {index_dir}")


def load_index(cfg: AppConfig) -> SimpleIndex:
    index_dir = Path(cfg.index.index_dir)
    if not index_dir.exists():
        raise FileNotFoundError(
            f"Index directory {index_dir} not found. "
            "Сначала запустите `python main.py build-index`."
        )

    emb_path = index_dir / "embeddings.npy"
    meta_path = index_dir / "chunks.jsonl"

    if not emb_path.exists() or not meta_path.exists():
        raise FileNotFoundError("Index files not found. Пересоберите индекс.")

    embeddings = np.load(emb_path)
    chunks: List[Dict] = []
    with open(meta_path, "r", encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))

    # BM25
    bm25 = None
    bm25_path = index_dir / "bm25.pkl"
    if bm25_path.exists():
        import pickle

        with open(bm25_path, "rb") as f:
            data = pickle.load(f)
            bm25 = data.get("bm25")

    # FAISS (если есть файл и backend не numpy)
    faiss_index = None
    backend = (cfg.index.vector_backend or "numpy").lower()
    faiss_path = index_dir / "faiss.index"
    if backend != "numpy" and faiss_path.exists():
        try:
            import faiss  # type: ignore
            faiss_index = faiss.read_index(str(faiss_path))
            print(f"🧱 Загружен FAISS индекс из {faiss_path}")
        except ImportError:
            print("⚠️  faiss не установлен, использую numpy backend.")

    return SimpleIndex(embeddings=embeddings, chunks=chunks, bm25=bm25, faiss_index=faiss_index)