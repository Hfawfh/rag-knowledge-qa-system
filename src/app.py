from __future__ import annotations

from pathlib import Path
from typing import Any

import gradio as gr

try:
    from .rag_pipeline import RAGPipeline
except ImportError:
    from rag_pipeline import RAGPipeline


pipeline = RAGPipeline()


def build_database(files: list[Any] | None) -> str:
    """根据上传的 PDF 文件建立向量知识库。"""

    if not files:
        return "请先上传至少一个 PDF 文件。"

    paths: list[str] = []

    for file in files:
        if isinstance(file, str):
            paths.append(file)
        elif hasattr(file, "name"):
            paths.append(str(Path(file.name)))
        else:
            return f"无法识别上传文件：{file}"

    try:
        chunk_count = pipeline.build_index(paths)
        return f"知识库建立完成，共生成 {chunk_count} 个文本块。"
    except Exception as exc:
        return f"建库失败：{exc}"


def convert_history_for_pipeline(
    history: list[dict[str, Any]] | None,
) -> list[tuple[str, str]]:
    """
    将 Gradio messages 格式转换成问答对格式，
    供原来的 pipeline.ask() 使用。
    """

    if not history:
        return []

    result: list[tuple[str, str]] = []
    user_message: str | None = None

    for item in history:
        if not isinstance(item, dict):
            continue

        role = item.get("role")
        content = item.get("content", "")

        if not isinstance(content, str):
            content = str(content)

        if role == "user":
            user_message = content

        elif role == "assistant" and user_message is not None:
            result.append((user_message, content))
            user_message = None

    return result


def chat(
    message: str,
    history: list[dict[str, Any]] | None,
) -> tuple[str, list[dict[str, str]]]:
    """处理用户问题并返回 Gradio messages 格式。"""

    message = (message or "").strip()
    history = list(history or [])

    if not message:
        return "", history

    # 只把本轮提问之前的历史传入 RAGPipeline
    pipeline_history = convert_history_for_pipeline(history)

    history.append(
        {
            "role": "user",
            "content": message,
        }
    )

    try:
        answer, sources = pipeline.ask(message, pipeline_history)

        answer_text = str(answer).strip()

        if sources:
            full_answer = (
                f"{answer_text}\n\n"
                f"### 检索来源\n"
                f"{str(sources).strip()}"
            )
        else:
            full_answer = answer_text

        history.append(
            {
                "role": "assistant",
                "content": full_answer,
            }
        )

    except Exception as exc:
        history.append(
            {
                "role": "assistant",
                "content": f"处理失败：{exc}",
            }
        )

    return "", history


def clear_chat() -> tuple[str, list[dict[str, str]]]:
    """清空输入框和聊天记录。"""

    return "", []


with gr.Blocks(title="RAG Knowledge QA System") as demo:
    gr.Markdown(
        "# RAG 智能知识库问答系统\n"
        "上传一个或多个 PDF，建立向量知识库后，"
        "即可进行带来源页码的问答。"
    )

    with gr.Row():
        files = gr.File(
            label="上传 PDF 文档",
            file_count="multiple",
            file_types=[".pdf"],
        )

        build_btn = gr.Button(
            "建立知识库",
            variant="primary",
        )

    build_status = gr.Textbox(
        label="建库状态",
        interactive=False,
    )

    build_btn.click(
        fn=build_database,
        inputs=files,
        outputs=build_status,
    )

    # Gradio 6 不需要、也不支持 type 参数
    chatbot = gr.Chatbot(
        label="知识库问答",
        height=480,
    )

    message = gr.Textbox(
        label="请输入问题",
        placeholder="例如：这篇论文的创新点是什么？",
        lines=2,
    )

    with gr.Row():
        submit_btn = gr.Button(
            "发送",
            variant="primary",
        )
        clear_btn = gr.Button("清空对话")

    message.submit(
        fn=chat,
        inputs=[message, chatbot],
        outputs=[message, chatbot],
    )

    submit_btn.click(
        fn=chat,
        inputs=[message, chatbot],
        outputs=[message, chatbot],
    )

    clear_btn.click(
        fn=clear_chat,
        inputs=None,
        outputs=[message, chatbot],
    )


if __name__ == "__main__":
    demo.launch()