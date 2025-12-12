from langchain_core.language_models import BaseLanguageModel
from transformers import PreTrainedModel, PreTrainedTokenizerBase
from pydantic import SecretStr
from model import LocalLoadModel
from model.RemoteLoadModel import RemoteLoadModel
from pathlib import Path
from typing import Dict, Optional, Tuple

ENV_PATH = Path(".env")


def save_env(env_data: Dict[str, str]) -> None:
    """保存配置到 .env 文件"""
    try:
        with open(ENV_PATH, mode="w", encoding='utf-8') as f:
            for key, value in env_data.items():
                # 简单处理：如果包含空格且没被引号包裹，则添加引号
                if ' ' in value and not (value.startswith('"') or value.startswith("'")):
                    f.write(f'{key}="{value}"\n')
                else:
                    f.write(f"{key}={value}\n")
        print(f"✅ 已保存配置到 {ENV_PATH}")
    except IOError as e:
        print(f"❌ 写入文件失败: {e}")


def load_env() -> Dict[str, str]:
    """读取 .env 文件"""
    if not ENV_PATH.exists():
        print("❌ 未找到 .env 文件")
        return {}

    env_vars = {}
    try:
        with open(ENV_PATH, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#') or '=' not in line:
                    continue
                key, value = line.split('=', 1)
                # 去除值两端的引号和空格
                env_vars[key.strip()] = value.strip().strip("'\"")
        return env_vars
    except IOError as e:
        print(f"❌ 读取文件失败: {e}")
        return {}


def get_input_config() -> Dict[str, str]:
    """引导用户输入配置"""
    print("\n请选择模式：\n1. 远程调用 (Key/Url/Secret)\n2. 本地调用 (Path)")

    while True:
        choice = input("请输入序号: ").strip()
        if choice == '1':
            return {
                "KEY": input("请输入 KEY: ").strip(),
                "URL": input("请输入 URL: ").strip(),
                "SECRET": input("请输入 SECRET: ").strip(),
                "MODEL_NAME": input("请输入 MODEL_NAME: ").strip(),
                "MODEL_PROVIDER": input("请输入 MODEL_PROVIDER (默认为openai): ").strip() or "openai"
            }
        elif choice == '2':
            return {
                "PATH": input("请输入 PATH: ").strip()
            }
        print("⚠️ 输入错误，请重新输入 1 或 2")


def identify_mode(config: Dict[str, str]) -> Tuple[Optional[str], str]:
    """
    校验配置并识别模式
    Returns: (Mode, Description)
    """
    if all(k in config and config[k] for k in ["KEY", "URL", "SECRET", "MODEL_NAME"]):
        return "REMOTE", "远程调用模型"
    elif all(k in config and config[k] for k in ["KEY", "URL", "MODEL_NAME"]):
        return "REMOTE", "远程调用模型"
    elif config.get("PATH"):
        return "LOCAL", "本地调用模型"
    return None, "❌ 配置无效或不完整"


def load() -> BaseLanguageModel | (PreTrainedModel, PreTrainedTokenizerBase):
    config = {}

    print("请选择配置来源：\n1. 读取本地 .env 文件\n2. 手动输入 (并保存)")
    while True:
        choice = input("请输入序号: ").strip()

        if choice == '1':
            config = load_env()
            if not config:  # 文件不存在或为空
                print("⚠️ 读取失败，转为手动输入模式...")
                config = get_input_config()
                save_env(config)
            break

        elif choice == '2':
            config = get_input_config()
            save_env(config)
            break

        print("⚠️ 输入错误，请重新输入")

    # 统一校验逻辑
    mode, message = identify_mode(config)
    print(f"\n当前状态: {message}")

    if mode == "REMOTE":
        # 在这里执行远程逻辑
        key = config.get("KEY")
        secret = config.get("SECRET", "")
        url = config.get("URL")
        model_name = config.get("MODEL_NAME")
        model_provider = config.get("MODEL_PROVIDER")
        # 使用RemoteLoadModel加载模型
        llm = RemoteLoadModel.load_model(
            api_key=SecretStr(key),
            api_secret=secret,
            base_url=url,
            model_name=model_name,
            model_provider=model_provider
        )
        print(f"执行远程逻辑 -> llm")
        return llm
    elif mode == "LOCAL":
        # 在这里执行本地逻辑
        path = config.get("PATH")
        # 拿到模型和分词器
        model, tokenizer = LocalLoadModel.load_model(path)
        print(f"执行本地逻辑 -> PATH")
        return model, tokenizer
    else:
        print("程序无法继续，请检查环境变量设置。")


if __name__ == "__main__":
    loaded_obj = load()
    if isinstance(loaded_obj, tuple):  # 本地调用逻辑
        model, tokenizer = loaded_obj
    elif hasattr(loaded_obj, 'invoke'):  # 远程调用逻辑 (LLM对象)
        llm = loaded_obj