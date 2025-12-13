import torch
from typing import List, Generator
from PIL import Image

# LangChain Core
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.pydantic_v1 import BaseModel, Field

# LangChain HuggingFace
from langchain_huggingface import ChatHuggingFace, HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Transformers & Diffusers (原生调用，用于特殊任务)
from transformers import pipeline
from diffusers import StableDiffusionPipeline

def gen_resp(llm, query):
    """
    基础模型生成请求 (基础补全模式)
    """
    # 简单的 Prompt 构造
    prompt = f"用户: {query}\n助手: "

    # invoke 返回的是字符串
    response = llm.invoke(prompt)

    # 有些模型生成时会包含输入的 prompt，这里做一个简单的切分处理（视情况而定）
    # 如果模型是 Chat 模型（如 Qwen），建议直接使用下面的 load_chat_model 方式
    return response

def gen_resp_with_chat(chat_model: ChatHuggingFace, query):
    """
    使用 ChatHuggingFace 生成响应
    你需要使用 ChatHuggingFace 创建实例
    """
    messages = [
        SystemMessage(content="你是一个人工智能领域专家，专门研究机器学习、深度学习和强化学习。"),
        HumanMessage(content=query)
    ]

    # invoke 返回的是 AIMessage 对象
    response = chat_model.invoke(messages)
    return response.content

def gen_stream_resp(chat_model: ChatHuggingFace, query: str, history: List = None) -> Generator[str, None, None]:
    """
    【流式响应】
    核心功能：逐字返回生成的 Token，用于前端打字机效果。
    适用场景：聊天机器人、长文本生成。
    """
    messages = [
        SystemMessage(content="你是一个人工智能领域专家，专门研究机器学习、深度学习和强化学习。"),
        *history,  # 注入历史对话
        HumanMessage(content=query)
    ]

    # 使用 .stream() 方法，这比 invoke 更适合实时交互
    try:
        for chunk in chat_model.stream(messages):
            # chunk.content 是当前生成的片段
            if chunk.content:
                yield chunk.content
    except Exception as e:
        yield f"生成出错: {str(e)}"

def gen_rag_resp(chat_model: ChatHuggingFace, query: str, documents: List[str]):
    """
    【RAG 检索增强响应】
    核心功能：基于提供的文档内容回答问题，减少幻觉。
    """
    # 1. 临时构建向量库 (实际生产中通常只构建一次并持久化)
    embeddings = chat_model.tokenizer()
    vectorstore = FAISS.from_texts(documents, embedding=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

    # 2. 定义 Prompt 模板
    template = """基于以下上下文回答用户的问题。如果上下文不包含答案，请说不知道。

    上下文: {context}

    问题: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)

    # 3. 构建 LCEL 链 (LangChain Expression Language)
    rag_chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | chat_model
    )

    return rag_chain.invoke(query).content

def gen_multimodal_resp(image_path: str, query: str):
    """
    【多模态/图片理解响应】
    注意：LangChain 对本地多模态模型的支持不如原生 Transformers 灵活。
    这里推荐直接使用 Transformers 的 pipeline 或专门的多模态模型（如 Qwen-VL, Llava）。
    """
    # 这里演示使用 transformers 原生 pipeline 做 Image-to-Text (VQA)
    # 你需要下载如 "llava-hf/llava-1.5-7b-hf" 或 "Qwen/Qwen-VL-Chat"

    # 假设使用一个轻量级的 VQA 模型
    vqa_pipeline = pipeline("visual-question-answering", model="dandelin/vilt-b32-finetuned-vqa", device=0)

    image = Image.open(image_path)
    result = vqa_pipeline(image=image, question=query)

    # 结果通常是 list of dict, 如 [{'score': 0.9, 'answer': 'cat'}]
    top_answer = result[0]['answer']
    return f"根据图片分析，答案可能是：{top_answer}"

def gen_image_generation_resp(prompt: str, output_path: str = "output.png"):
    """
    【图像生成响应】
    核心功能：文生图 (Text-to-Image)。
    依赖：diffusers 库
    """
    # 首次加载会比较慢，建议在服务启动时全局加载
    model_id = "stabilityai/stable-diffusion-2-1-base"
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe = pipe.to("cuda")

    image = pipe(prompt).images[0]
    image.save(output_path)
    return f"图片已生成并保存至: {output_path}"

# --- 3. 高级功能：结构化输出 ---

class AnalysisResult(BaseModel):
    sentiment: str = Field(description="用户情绪，例如：积极、消极、中性")
    keywords: List[str] = Field(description="提取的3个关键实体")
    summary: str = Field(description="一句话总结用户意图")

def gen_structured_resp(chat_model: ChatHuggingFace, query: str) -> dict:
    """
    【结构化 JSON 响应】
    核心功能：强制模型输出 JSON 格式，方便后端程序处理。
    """
    parser = JsonOutputParser(pydantic_object=AnalysisResult)

    prompt = PromptTemplate(
        template="请分析以下文本。\n{format_instructions}\n文本: {query}",
        input_variables=["query"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    chain = prompt | chat_model | parser

    # 返回的是 Python 字典，可以直接 json.dumps 给前端
    result = chain.invoke({"query": query})
    return result