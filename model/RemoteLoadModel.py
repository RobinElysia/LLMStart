import requests
import json
from typing import Dict, Iterator

def gen_resp_stream(config: Dict, query: str) -> Iterator[str]:
    """
    流式生成请求
    :param config: 配置字典
    :param query: 查询提示词
    :yield: 模型生成的 str 片段
    """
    # 构建消息列表
    messages = [
        {"role": "system", "content": "你是一个人工智能领域专家，专门研究机器学习、深度学习和强化学习"},
        {"role": "user", "content": query}
    ]

    # 准备请求数据，添加stream=True
    payload = {
        "model": config['model_name'],
        "messages": messages,
        "max_tokens": 512,
        "temperature": 0.7,
        "stream": True
    }

    try:
        # 发送流式请求
        response = requests.post(
            f"{config['base_url']}/chat/completions",
            headers=config['headers'],
            json=payload,
            timeout=30,
            stream=True
        )

        if response.status_code != 200:
            yield f"请求失败: {response.status_code}, {response.text}"
            return

        # 解析流式响应
        buffer = ""
        for line in response.iter_lines():
            if line:
                line = line.decode('utf-8')
                if line.startswith('data: '):
                    data = line[6:]  # 去掉 'data: '
                    if data == '[DONE]':
                        break
                    try:
                        chunk = json.loads(data)
                        if 'choices' in chunk and chunk['choices']:
                            delta = chunk['choices'][0].get('delta', {})
                            content = delta.get('content', '')
                            if content:
                                yield content
                    except json.JSONDecodeError:
                        continue

    except Exception as e:
        yield f"请求异常: {str(e)}"

# if __name__ == "__main__":
#     # 配置参数
#     config = load_model(API_KEY, API_SECRET, BASE_URL, MODEL_NAME)
#
#     # 流式调用
#     query = "什么是机器学习？"
#     for chunk in gen_resp_stream(config, query):
#         print(chunk, end="", flush=True)
#
#     # 或者收集完整的响应
#     full_response = ""
#     for chunk in gen_resp_stream(config, query):
#         full_response += chunk