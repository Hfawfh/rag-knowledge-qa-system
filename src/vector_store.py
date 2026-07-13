from __future__ import annotations

from pathlib import Path
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

try:
    from .config import RAGConfig
except ImportError:
    from config import RAGConfig


class VectorStoreManager:
    def __init__(self, config: RAGConfig) -> None:
        self.config = config
        self.embeddings = HuggingFaceBgeEmbeddings(
            model_name=config.embedding_model,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        self.db: FAISS | None = None

    def build(self, documents: list) -> int:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
        )
        chunks = splitter.split_documents(documents)
        if not chunks:
            raise ValueError("文档未提取到有效文本。")

        self.db = FAISS.from_documents(chunks, self.embeddings)
        Path(self.config.vector_store_path).mkdir(parents=True, exist_ok=True)
        self.db.save_local(self.config.vector_store_path)
        return len(chunks)

    def load(self) -> FAISS:
        if self.db is None:
            path = Path(self.config.vector_store_path)
            if not path.exists():
                raise FileNotFoundError("向量库不存在，请先上传文档并建立知识库。")
            self.db = FAISS.load_local(
                str(path),
                self.embeddings,
                allow_dangerous_deserialization=True,
            )
        return self.db

    def search(self, query: str, top_k: int | None = None) -> list:
        db = self.load()
        return db.similarity_search(query, k=top_k or self.config.top_k)