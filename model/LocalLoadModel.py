from langchain_core.language_models import BaseLLM
from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_core.messages import HumanMessage, SystemMessage
import torch


def load_model(model_path: str) -> BaseLLM:
    """
    创建模型
    :param model_path: 模型本地路径
    :return: 返回 LangChain LLM 实例
    """
    # 1. 加载原始模型和分词器
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",  # 推荐使用 auto，或者根据实际情况指定 {"": 0}
        torch_dtype=torch.bfloat16,
        local_files_only=True,
        trust_remote_code=True
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=True
    )

    # 2. 创建原生的 transformers pipeline
    # 注意：生成的参数（temperature, max_new_tokens等）在这里设置
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,
        do_sample=True,
        temperature=0.7,
        pad_token_id=tokenizer.eos_token_id,
        # device=0 # 如果前面模型加载使用了 device_map，这里通常不需要再指定 device，或者设为 None
    )

    # 3. 实例化 langchain_huggingface 的 HuggingFacePipeline
    llm = HuggingFacePipeline(pipeline=pipe)

    return llm


def gen_resp(llm, query):
    """
    生成请求 (基础补全模式)
    """
    # 简单的 Prompt 构造
    prompt = f"用户: {query}\n助手: "

    # invoke 返回的是字符串
    response = llm.invoke(prompt)

    # 有些模型生成时会包含输入的 prompt，这里做一个简单的切分处理（视情况而定）
    # 如果模型是 Chat 模型（如 Qwen），建议直接使用下面的 load_chat_model 方式
    return response


def load_chat_model(model_path: str):
    """
    使用 ChatHuggingFace 加载对话模型 (推荐用于 Qwen 等对话模型)
    """
    # 复用上面的逻辑加载基础 llm
    llm = load_model(model_path)

    # 包装成 Chat 模型
    chat_model = ChatHuggingFace(llm=llm)

    return chat_model


def gen_resp_with_chat(chat_model, query):
    """
    使用 ChatHuggingFace 生成响应
    """
    messages = [
        SystemMessage(content="你是一个人工智能领域专家，专门研究机器学习、深度学习和强化学习。"),
        HumanMessage(content=query)
    ]

    # invoke 返回的是 AIMessage 对象
    response = chat_model.invoke(messages)
    return response.content

# if __name__ == "__main__":
#     # 请确保路径正确，且不包含中文字符以免引发某些编码问题
#     model_path = r"D:\code\python\LearnPyLib\model\Generate\Qwen2.5 0.5b"
#
#     # --- 方式 1: 基础文本生成 (Text Completion) ---
#     print("正在加载模型...")
#     llm = load_model(model_path)
#     query = "请解释一下Transformer的工作原理"
#     print(f"\n--- Base Query: {query} ---")
#     response = gen_resp(llm, query)
#     print("Response:", response)
#
#     # --- 方式 2: 对话模式 (Chat Completion) ---
#     # 如果你使用的是 Qwen, Llama3-Instruct 等指令微调模型，强烈建议使用这种方式
#     # 因为它会自动应用模型的 Chat Template
#     print("\n--- Switching to Chat Mode ---")
#     chat_model = ChatHuggingFace(llm=llm)  # 可以直接复用上面的 llm
#     chat_resp = gen_resp_with_chat(chat_model, query)
#     print("Chat Response:", chat_resp)