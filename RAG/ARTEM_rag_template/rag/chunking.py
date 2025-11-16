from typing import List, Dict
import re

from transformers import AutoTokenizer

from .config import AppConfig, ChunkingConfig


def _chunk_char(
    text: str,
    doc_id: int,
    source: str,
    cfg: ChunkingConfig,
) -> List[Dict]:
    max_chars = cfg.max_chars
    overlap_chars = int(max_chars * cfg.overlap_ratio)
    overlap_chars = max(1, min(overlap_chars, max_chars - 1))

    chunks = []
    start = 0
    chunk_id = 0
    length = len(text)

    while start < length:
        end = min(start + max_chars, length)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(
                {
                    "doc_id": doc_id,
                    "source": source,
                    "chunk_id": chunk_id,
                    "text": chunk,
                }
            )
            chunk_id += 1
        start = end - overlap_chars
        if start < 0:
            start = 0
        if start >= length:
            break

    return chunks


def _split_into_sentences(text: str) -> List[str]:
    # простая разбивка по .?!…
    text = text.replace("\n", " ").strip()
    if not text:
        return []
    sentences = re.split(r"(?<=[\.!?…])\s+", text)
    sentences = [s.strip() for s in sentences if s.strip()]
    return sentences


def _chunk_semantic(
    text: str,
    doc_id: int,
    source: str,
    cfg: ChunkingConfig,
) -> List[Dict]:
    max_chars = cfg.max_chars
    overlap_ratio = cfg.overlap_ratio
    min_chars = cfg.sentence_min_chars

    sentences = _split_into_sentences(text)
    if not sentences:
        return []

    chunks: List[Dict] = []
    current: List[str] = []
    cur_len = 0
    chunk_id = 0

    for sent in sentences:
        s = sent.strip()
        if not s:
            continue

        # если чанка ещё мало, добавляем даже если будет > max_chars
        if cur_len + len(s) + 1 <= max_chars or cur_len < min_chars:
            current.append(s)
            cur_len += len(s) + 1
            continue

        # закрываем текущий чанк
        chunk_text = " ".join(current).strip()
        if chunk_text:
            chunks.append(
                {
                    "doc_id": doc_id,
                    "source": source,
                    "chunk_id": chunk_id,
                    "text": chunk_text,
                }
            )
            chunk_id += 1

        # overlap по предложениям
        if current:
            n_overlap = max(1, int(len(current) * overlap_ratio))
            overlap_sents = current[-n_overlap:]
        else:
            overlap_sents = []

        current = overlap_sents + [s]
        cur_len = sum(len(x) + 1 for x in current)

    # финальный чанк
    if current:
        chunk_text = " ".join(current).strip()
        if chunk_text:
            chunks.append(
                {
                    "doc_id": doc_id,
                    "source": source,
                    "chunk_id": chunk_id,
                    "text": chunk_text,
                }
            )

    return chunks


# ---------- НОВОЕ: токенный чанкинг ----------

_tokenizers_cache = {}


def _get_tokenizer(model_name: str):
    tok = _tokenizers_cache.get(model_name)
    if tok is None:
        tok = AutoTokenizer.from_pretrained(model_name)
        _tokenizers_cache[model_name] = tok
    return tok


def _chunk_token(
    cfg: AppConfig,
    text: str,
    doc_id: int,
    source: str,
) -> List[Dict]:
    """
    Токенный чанкинг:
      - используем токенайзер из cfg.chunking.tokenizer_model
        или cfg.embedding.model_name;
      - max_tokens / overlap_tokens берём из cfg.chunking,
        либо считаем из max_chars/overlap_ratio.
    """
    c = cfg.chunking

    # модель токенайзера
    tokenizer_model = c.tokenizer_model or cfg.embedding.model_name
    tok = _get_tokenizer(tokenizer_model)

    # параметры
    max_tokens = c.max_tokens or c.max_chars
    overlap_tokens = c.overlap_tokens or int(max_tokens * c.overlap_ratio)
    overlap_tokens = max(1, min(overlap_tokens, max_tokens - 1))

    # токенизируем
    input_ids = tok.encode(text, add_special_tokens=False)
    n = len(input_ids)
    if n == 0:
        return []

    chunks: List[Dict] = []
    start = 0
    chunk_id = 0

    while start < n:
        end = min(start + max_tokens, n)
        chunk_ids = input_ids[start:end]
        chunk_text = tok.decode(chunk_ids, skip_special_tokens=True).strip()
        if chunk_text:
            chunks.append(
                {
                    "doc_id": doc_id,
                    "source": source,
                    "chunk_id": chunk_id,
                    "text": chunk_text,
                }
            )
            chunk_id += 1

        if end == n:
            break
        start = end - overlap_tokens
        if start < 0:
            start = 0

    return chunks


def chunk_document(
    cfg: AppConfig,
    text: str,
    doc_id: int,
    source: str,
) -> List[Dict]:
    """
    Унифицированный вход: выбирает стратегию чанкинга по cfg.chunking.type.

    - "semantic" → _chunk_semantic
    - "token"    → токенный чанкинг по transformers
    - прочее     → посимвольный
    """
    c = cfg.chunking
    t = (c.type or "semantic").lower()

    if t == "semantic":
        return _chunk_semantic(text, doc_id=doc_id, source=source, cfg=c)
    if t == "token":
        return _chunk_token(cfg, text=text, doc_id=doc_id, source=source)

    # fallback — старый посимвольный
    return _chunk_char(text, doc_id=doc_id, source=source, cfg=c)