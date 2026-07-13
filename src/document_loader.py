from __future__ import annotations

from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader


def load_pdf_documents(paths: list[str]) -> list:
    documents = []
    for raw_path in paths:
        path = Path(raw_path)
        if path.suffix.lower() != ".pdf":
            raise ValueError(f"仅支持 PDF 文件: {path}")
        if not path.exists():
            raise FileNotFoundError(f"文件不存在: {path}")

        pages = PyPDFLoader(str(path)).load()
        for page in pages:
            page.metadata["source"] = path.name
        documents.extend(pages)
    return documents