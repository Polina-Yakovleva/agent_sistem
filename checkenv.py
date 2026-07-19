"""Validate runtime environment variables for the agent package.

Usage:
    python checkenv.py           # validate only
    python checkenv.py --init    # create .env from defaults if missing, then validate
"""

import os
import sys

SECRET_KEYS = (
    "LLM_BASE_URL",
    "LLM_API_KEY",
    "PG_PASSWORD",
    "LANGFUSE_PUBLIC_KEY",
    "LANGFUSE_SECRET_KEY",
)

REQUIRED_KEYS = (
    "LLM_MODEL",
    "LLM_API_KEY",
    "PG_HOST",
    "PG_PORT",
    "PG_DATABASE",
    "PG_USER",
    "PG_PASSWORD",
    "QDRANT_HOST",
    "QDRANT_PORT",
    "QDRANT_COLLECTION",
)

env = {
    "LLM_BASE_URL": "",  # !!! SECRET
    "LLM_MODEL": "hf.co/NightForger/avibe-GGUF:Q4_K_M",
    "LLM_API_KEY": "",  # !!! SECRET
    "LLM_TEMPERATURE": "0.1",
    "LLM_MAX_TOKENS": "1024",
    "LLM_TIMEOUT": "300",
    "PG_HOST": "localhost",
    "PG_PORT": "5432",
    "PG_DATABASE": "airport",
    "PG_USER": "postgres",
    "PG_PASSWORD": "",  # !!! SECRET
    "CHECKPOINT_BACKEND": "postgres",
    "QDRANT_HOST": "localhost",
    "QDRANT_PORT": "6333",
    "QDRANT_COLLECTION": "airline_BAAI_bge-m3",
    "EMBEDDING_MODEL": "BAAI/bge-m3",
    "EMBEDDING_DEVICE": "cpu",
    "RAG_TOP_K": "3",
    "AGENT_USER_ID": "default",
    "MEMORY_ENABLED": "true",
    "EVAL_MODE": "false",
    "MAX_REVISIONS": "2",
    "OBSERVABILITY_ENABLED": "true",
    "LOG_LEVEL": "INFO",
    "LOG_FORMAT": "json",
    "METRICS_ENABLED": "true",
    "METRICS_PORT": "9100",
    "LANGFUSE_ENABLED": "false",
    "LANGFUSE_HOST": "http://localhost:3000",
    "LANGFUSE_PUBLIC_KEY": "",  # !!! SECRET
    "LANGFUSE_SECRET_KEY": "",  # !!! SECRET
}


def _load_dotenv(dotenv_path):
    if not os.path.exists(dotenv_path):
        return
    with open(dotenv_path, encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


def _write_dotenv(dotenv_path):
    lines = [f"{key}={value}" for key, value in env.items()]
    with open(dotenv_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    init = "--init" in argv

    root = os.path.dirname(os.path.abspath(__file__))
    dotenv_path = os.path.join(root, ".env")

    if init:
        if not os.path.exists(dotenv_path):
            _write_dotenv(dotenv_path)
            print("Created .env from defaults in checkenv.py. Fill required secrets and run again.")
            return 1
        print(".env already exists; skip --init write.")

    if os.path.exists(dotenv_path):
        _load_dotenv(dotenv_path)

    # Переопределяем хардкод значениями из переменных окружения / .env
    for name in list(env.keys()):
        if os.environ.get(name) is not None:
            env[name] = os.environ.get(name)

    print("Визуальный контроль параметров приложения:")
    for name, value in env.items():
        if name in SECRET_KEYS:
            print(f"  {name} (количество символов в значении параметра: {len(str(value))})")
        else:
            print(f"  {name}={value}")

    check = True
    print("Контроль допустимости параметров приложения")
    for key in REQUIRED_KEYS:
        value = env.get(key)
        if value is None or str(value).strip() == "":
            print(f"  Параметр {key} — пустой или отсутствует")
            check = False
        else:
            print(f"  Параметр {key} — OK")

    print("Контроль допустимости параметров приложения завершён")

    if not check:
        print("Из-за ошибок валидации параметров запуск невозможен.")
        if not init and not os.path.exists(dotenv_path):
            print("Подсказка: python checkenv.py --init")
        return 1

    print("Environment check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
