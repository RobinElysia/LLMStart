from typing import Any
from pydantic import SecretStr
from langchain_core.language_models import BaseLanguageModel
from langchain_openai import ChatOpenAI

def load_model(
        api_key: SecretStr,
        api_secret: str,
        base_url: str,
        model_name: str,
        model_provider: str = "openai",
        **kwargs: Any
) -> BaseLanguageModel:
    """
    创建智能体对象

    Args:
        api_key: API密钥
        api_secret: API密钥
        base_url: API基础URL
        model_name: 模型名称
        model_provider: 模型提供商 ('openai', 'zhipuai', 'dashscope', 'qwen', 'moonshot' 等)
        **kwargs: 其他参数

    Returns:
        BaseLanguageModel: 语言模型实例
    """
    # 根据不同的模型提供商创建相应的模型实例
    if model_provider.lower() == "openai":
        # OpenAI 或兼容 OpenAI API 的模型服务
        llm = ChatOpenAI(
            model=model_name,
            base_url=base_url,
            api_key=api_key,
            **kwargs
        )
    elif model_provider.lower() == "zhipuai":
        # 智谱AI
        try:
            from langchain_community.llms import ZhipuAI
            llm = ZhipuAI(
                model=model_name,
                api_key=api_key,
                **kwargs
            )
        except ImportError:
            raise ImportError("请安装 langchain-community 来使用智谱AI模型")
    elif model_provider.lower() == "dashscope":
        # 阿里云百炼平台(通义千问)
        try:
            from langchain_community.llms import Tongyi
            llm = Tongyi(
                model_name=model_name,
                dashscope_api_key=api_key,
                **kwargs
            )
        except ImportError:
            raise ImportError("请安装 langchain-community 来使用阿里云通义千问模型")
    elif model_provider.lower() == "qwen":
        # 阿里云百炼平台(通义千问) - 另一种方式
        llm = ChatOpenAI(
            model=model_name,
            base_url=base_url or "https://dashscope.aliyuncs.com/api/v1",
            api_key=api_key,
            **kwargs
        )
    elif model_provider.lower() == "moonshot":
        # Moonshot AI (月之暗面)
        llm = ChatOpenAI(
            model=model_name,
            base_url=base_url or "https://api.moonshot.cn/v1",
            api_key=api_key,
            **kwargs
        )
    else:
        # 默认使用 OpenAI 兼容方式
        llm = ChatOpenAI(
            model=model_name,
            base_url=base_url,
            api_key=api_key,
            **kwargs
        )
    return llm