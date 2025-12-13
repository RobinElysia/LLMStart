import torch
from typing import Optional, List, Any
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoModelForVision2Seq,
    AutoProcessor,
    pipeline,
)
from langchain_huggingface import HuggingFacePipeline
from langchain_core.language_models.llms import LLM, BaseLLM
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from pydantic import Field


class LangChainMultimodalWrapper(LLM):
    """
    自定义包装器，用于将文生图或图生文模型伪装成 LangChain 的 LLM。
    - 对于文生图：输入 Prompt，保存图片，返回"图片已保存至..."的文本。
    - 对于图生文：处理输入中的图片路径或URL（需要特定的 Prompt 格式）。
    """
    pipeline: Any = Field(description="HuggingFace Pipeline object")
    task: str = Field(description="Model task type")

    def _call(
            self,
            prompt: str,
            stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
            **kwargs: Any,
    ) -> str:
        # 1. 文生图任务 (Text-to-Image)
        if self.task == "text-to-image":
            # 调用管道生成图片
            image = self.pipeline(prompt).images[0]
            # 为了让LangChain链路不中断，我们需要返回字符串
            # 实际生产中，这里可以将图片上传云端并返回URL，或者保存本地
            save_path = f"generated_image_{hash(prompt)}.png"
            image.save(save_path)
            return f"[System]: Image generated and saved to {save_path}"

        # 2. 图生文任务 (Image-to-Text / VQA)
        elif self.task == "image-to-text":
            # 假设 Prompt 格式为 "image_path|question" 或者直接传入图片URL
            # 这里做一个简单的解析逻辑
            if "|" in prompt:
                img_input, text_input = prompt.split("|", 1)
            else:
                # 如果没有分隔符，默认 prompt 是图片路径，只做描述
                img_input = prompt
                text_input = None

            # 调用多模态管道
            if text_input:
                result = self.pipeline(img_input, text_input)
            else:
                result = self.pipeline(img_input)

            # 提取结果文本
            return str(result[0]['generated_text'])

        return "Unsupported task for this wrapper."

    @property
    def _llm_type(self) -> str:
        return "custom_multimodal_wrapper"

def load_model(
        model_path: str,
        task: str = "text-generation", # ('text-generation', 'text2text-generation', 'image-to-text', 'text-to-image')
        model_type: str = "causal",  # causal, seq2seq, vision, diffusion
        temperature: float = 0.7,
        torch_dtype: torch.dtype = torch.float16
) -> BaseLLM:
    """
    使用 Transformers 动态加载模型并转化为 LangChain 对象。

    Args:
        model_path: 本地模型路径或 HuggingFace ID
        task: 任务类型 ('text-generation', 'text2text-generation', 'image-to-text', 'text-to-image')
        model_type: 模型架构提示 ('causal' for GPT, 'seq2seq' for T5, 'vision' for LLaVA, 'diffusion' for SD)
        temperature: 生成温度
        torch_dtype: 模型精度 (torch.float16, torch.bfloat16 等)

    Returns:
        LangChain Runnable / BaseLLM 对象
    """

    print(f"Loading model from {model_path} for task: {task}...")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_kwargs = {"torch_dtype": torch_dtype}

    # 1. 根据 model_type 加载对应的 Model 和 Tokenizer/Processor
    try:
        if model_type == "causal":
            # 适用于 Llama, Qwen, Mistral 等
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map="auto",
                **model_kwargs
            )
        elif model_type == "seq2seq":
            # 适用于 T5, FLAN 等
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoModelForSeq2SeqLM.from_pretrained(
                model_path,
                device_map="auto",
                **model_kwargs
            )
        elif model_type == "vision":
            # 适用于 LLaVA, Qwen-VL, Git 等 (图生文)
            # 注意：多模态模型通常使用 Processor 而不是单纯的 Tokenizer
            tokenizer = AutoProcessor.from_pretrained(model_path)  # 这里使用 Processor
            model = AutoModelForVision2Seq.from_pretrained(
                model_path,
                device_map="auto",
                **model_kwargs
            )
        elif model_type == "diffusion":
            # 适用于 Stable Diffusion (文生图)
            # Diffusion 模型加载逻辑略有不同，通常直接用 pipeline 加载更方便
            # 这里为了统一，我们留空，直接在下方 pipeline 阶段处理
            tokenizer = None
            model = None
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        # 2. 创建 Transformers Pipeline
        if model_type == "diffusion":
            # 文生图专用 Pipeline
            pipe = pipeline("text-to-image", model=model_path, torch_dtype=torch_dtype, device_map="auto")
        else:
            # 通用 Pipeline
            pipe = pipeline(
                task=task,
                model=model,
                tokenizer=tokenizer,  # 对于 vision 模型，这里传入 processor 可能会有 warning，但 pipeline 通常能自动处理
                image_processor=tokenizer if model_type == "vision" else None,
                device_map="auto",
                max_new_tokens=512,
                model_kwargs={"temperature": temperature} if temperature > 0 else {}
            )

        # 3. 转化为 LangChain 对象

        # 情况 A: 标准文本生成 (LangChain 原生支持最好)
        if task in ["text-generation", "text2text-generation"]:
            lc_model = HuggingFacePipeline(pipeline=pipe)
            print("Loaded as Standard HuggingFacePipeline")
            return lc_model

        # 情况 B: 多模态任务 (需要自定义 Wrapper 才能在 Chain 中流转)
        elif task in ["image-to-text", "text-to-image"]:
            lc_model = LangChainMultimodalWrapper(pipeline=pipe, task=task)
            print(f"Loaded as Custom Multimodal Wrapper for {task}")
            return lc_model

        else:
            raise ValueError(f"Unsupported task for integration: {task}")

    except Exception as e:
        print(f"Error loading model: {e}")
        raise e