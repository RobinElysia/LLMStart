from typing import Optional

from langchain_core.language_models import BaseLanguageModel


def gen_resp(
        llm: BaseLanguageModel,
        query: str,
        system_message: Optional[str] = "你是一个人工智能领域专家，专门研究机器学习、深度学习和强化学习。"
) -> str:
    """
    生成响应

    Args:
        llm: 语言模型实例
        query: 用户查询
        system_message: 系统消息（可选）

    Returns:
        str: 模型生成的响应（基本响应）
    """
    from langchain_core.messages import HumanMessage, SystemMessage
    messages = []
    if system_message:
        messages.append(SystemMessage(content=system_message))
    else:
        messages.append(SystemMessage(content="你是一个人工智能领域专家，专门研究机器学习、深度学习和强化学习。"))
    messages.append(HumanMessage(content=query))
    response = llm.invoke(messages)
    return response.content