from __future__ import annotations

from langchain_openai import ChatOpenAI

try:
    from .config import RAGConfig
except ImportError:
    from config import RAGConfig


class LLMClient:
    def __init__(self, config: RAGConfig) -> None:
        if not config.api_key:
            raise ValueError("未配置 OPENAI_API_KEY，请复制 .env.example 为 .env。")

        self.client = ChatOpenAI(
            model=config.llm_model,
            api_key=config.api_key,
            base_url=config.base_url,
            temperature=config.temperature,
        )

    def answer(self, question: str, documents: list, history: list | None = None) -> str:
        context_parts = []
        for i, doc in enumerate(documents, start=1):
            source = doc.metadata.get("source", "unknown")
            page = int(doc.metadata.get("page", 0)) + 1
            context_parts.append(
                f"[资料{i} | 文件:{source} | 页码:{page}]\n{doc.page_content}"
            )

        history_text = ""
        if history:
            recent = history[-6:]
            history_text = "\n".join(
                f"用户:{item[0]}\n助手:{item[1]}" for item in recent
            )

        prompt = f"""
你是一个基于私有知识库回答问题的专业助手。

要求：
1. 仅依据提供的资料回答。
2. 资料不足时明确说明“当前资料中未找到足够依据”。
3. 不要编造论文结论、数字或实验结果。
4. 回答结尾列出引用来源，格式为：文件名，第X页。

对话历史：
{history_text}

检索资料：
{chr(10).join(context_parts)}

用户问题：
{question}
"""
        return self.client.invoke(prompt).content