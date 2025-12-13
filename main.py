from model.InitModel import load
from fastapi import FastAPI

app = FastAPI()

if __name__ == "__main__":
    llm = load() # 加载模型