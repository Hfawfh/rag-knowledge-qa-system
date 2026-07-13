from __future__ import annotations

try:
    from .config import load_config
    from .document_loader import load_pdf_documents
    from .vector_store import VectorStoreManager
    from .llm_client import LLMClient
except ImportError:
    from config import load_config
    from document_loader import load_pdf_documents
    from vector_store import VectorStoreManager
    from llm_client import LLMClient


class RAGPipeline:
    def __init__(self, config_path: str = "configs/config.yaml") -> None:
        self.config = load_config(config_path)
        self.store = VectorStoreManager(self.config)
        self._llm: LLMClient | None = None

    @property
    def llm(self) -> LLMClient:
        if self._llm is None:
            self._llm = LLMClient(self.config)
        return self._llm

    def build_index(self, file_paths: list[str]) -> int:
        documents = load_pdf_documents(file_paths)
        self.store.db = None
        return self.store.build(documents)

    def ask(self, question: str, history: list | None = None) -> tuple[str, str]:
        if not question.strip():
            return "请输入问题。", ""

        docs = self.store.search(question)
        answer = self.llm.answer(question, docs, history)

        sources = []
        for doc in docs:
            source = doc.metadata.get("source", "unknown")
            page = int(doc.metadata.get("page", 0)) + 1
            preview = doc.page_content[:160].replace("\n", " ")
            sources.append(f"- {source} 第{page}页：{preview}...")

        return answer, "\n".join(sources)