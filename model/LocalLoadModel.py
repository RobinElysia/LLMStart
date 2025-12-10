import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def load_model(model_path: str):
    """
    创建模型
    :param model_path: 模型本地路径
    :return: 返回 model 实例，tokenizer 实例
    """
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map={"":0},
        torch_dtype=torch.bfloat16,
        local_files_only=True,
        trust_remote_code=True
    )
    model.eval() # 评估模式
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=True
    )
    return model, tokenizer


def gen_resp(model, tokenizer, query):
    """
    生成请求
    :param model: 模型实例
    :param tokenizer: 模型分词器
    :param query: 查询提示词
    :return: 模型生成的 str
    """
    # 构建消息列表
    messages = [
        {"role": "system", "content": "你是一个人工智能领域专家，专门研究机器学习、深度学习和强化学习"},
        {"role": "user", "content": query}
    ]
    # 应用聊天模板
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    # 字符串转张量
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    # 生成
    with torch.no_grad():
        # generated_ids是一个二维的[1,n]的张量，n包含问题+答案，假设答案长度是 l
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.7
        )
    # 解码输出（跳过输入部分）
    # model_inputs.input_ids.shape[1]这是问题，截出后面是答案
    response_ids = generated_ids[:, model_inputs.input_ids.shape[1]:]
    # 截的答案仍是二维[1,l]，需要转成1维
    response = tokenizer.decode(response_ids[0], skip_special_tokens=True)
    return response

# if __name__ == "__main__":
#     model, tokenizer = load_model()
#     query = "北京有什么好玩的？"
#     print(gen_resp(model, tokenizer, query))