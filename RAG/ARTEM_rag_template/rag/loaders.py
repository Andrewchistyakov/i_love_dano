# rag/loaders.py
from pathlib import Path
from typing import List, Dict, Optional

from pypdf import PdfReader


def _read_pdf(path: Path) -> str:
    """
    Простое извлечение текста из PDF.
    Работает только с PDF, где есть текстовый слой.
    Для сканов нужна отдельная OCR-пайплайн.
    """
    reader = PdfReader(str(path))
    texts = []
    for page in reader.pages:
        try:
            t = page.extract_text() or ""
        except Exception:
            t = ""
        if t:
            texts.append(t)
    return "\n\n".join(texts).strip()


def load_documents(docs_path: str) -> List[Dict]:
    """
    Загружает все .txt / .md / .pdf файлы из папки как документы.
    Возвращает список dict:
      { "id": int, "path": str, "text": str }
    """
    base = Path(docs_path)
    if not base.exists():
        raise FileNotFoundError(f"Docs path not found: {docs_path}")

    docs: List[Dict] = []
    doc_id = 0

    exts = ("*.txt", "*.md", "*.pdf")

    for ext in exts:
        for p in base.rglob(ext):
            text: Optional[str] = None

            if p.suffix.lower() in {".txt", ".md"}:
                text = p.read_text(encoding="utf-8", errors="ignore")
            elif p.suffix.lower() == ".pdf":
                text = _read_pdf(p)

            if not text:
                # пустые/нечитаемые pdf можно просто пропустить или логировать
                continue

            docs.append(
                {
                    "id": doc_id,
                    "path": str(p),
                    "text": text,
                }
            )
            doc_id += 1

    if not docs:
        raise RuntimeError(f"No supported files (.txt/.md/.pdf) found in {docs_path}")

    return docs