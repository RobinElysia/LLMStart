# LLMStart
本项目是一个高性能的Python后端服务，采用FastAPI、Transformers和FAISS构建，旨在实现从Java后端系统无缝远程推断模型。
该模块提供了一个可扩展的RESTful API，用于提供高效的模型推理服务。

## 技术栈
- Python 3.11
- FastAPI
- Transformers
- FAISS
- Docker

## 技术实现
入口文件[main.py](main.py)是基本的，这个文件会询问你基本的配置信息，比如是基于远程调用API还是本地模型。（√）

## 快速开始
1. clone本项目到本地
2. 确保你的本地环境里有uv，之后执行`uv sync`
3. （可选步骤）创建`.env`文件，填入必要参数：
```bash
# 远程配置
KEY="YOUR_API_KEY"
SECRET="YOUR_API_SECRET"
URL="URL"

# 本地远程二选一

# 本地配置
PATH="PATH" # 模型地址（指定到文件夹即可）
```
4. 运行[main.py](main.py)即可

## 贡献
[RobinElysia](https://elysia.wiki:223/AboutUs.html)