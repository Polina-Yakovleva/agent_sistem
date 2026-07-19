"""Централизованная конфигурация системы.

Значения берутся из переменных окружения (или файла .env), при отсутствии
используются разумные значения по умолчанию для локального запуска.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- PostgreSQL ---
    pg_host: str = "localhost"
    pg_port: int = 5432
    pg_database: str = "airport"
    pg_user: str = "postgres"
    pg_password: str = "admin"

    # Вместимость самолёта по умолчанию (в схеме нет колонки capacity,
    # поэтому доступность мест считается относительно этого значения).
    flight_seat_capacity: int = 180

    # --- Qdrant / RAG ---
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_collection: str = "airline_BAAI_bge-m3"

    embedding_model: str = "BAAI/bge-m3"
    embedding_device: str = "cpu"

    # Сколько чанков возвращать из векторного поиска по умолчанию.
    rag_top_k: int = 3

    # --- External Agent (погода и отели через внешние API без ключей) ---
    open_meteo_forecast_url: str = "https://api.open-meteo.com/v1/forecast"
    open_meteo_geocoding_url: str = "https://geocoding-api.open-meteo.com/v1/search"
    # Photon (komoot) — бесключевой геокодер и POI-поиск по данным OSM.
    photon_url: str = "https://photon.komoot.io/api/"

    # Описательный User-Agent (правила использования публичных OSM-сервисов).
    # Значение должно быть ASCII (требование HTTP-заголовков).
    external_user_agent: str = "airline-agent-system/1.0 (VKR thesis; contact: student@example.com)"

    hotel_radius_km: float = 3.0  # радиус поиска отелей по умолчанию
    hotel_limit: int = 5  # максимум отелей в ответе
    weather_forecast_days: int = 1  # горизонт прогноза по умолчанию
    external_timeout: int = 15  # таймаут HTTP-запросов к внешним API, сек

    # --- LLM (A-vibe через Ollama, OpenAI-совместимый API) ---
    # Модель поднимается командой: ollama run hf.co/NightForger/avibe-GGUF:Q4_K_M
    llm_base_url: str = "http://localhost:11434/v1"
    llm_model: str = "hf.co/NightForger/avibe-GGUF:Q4_K_M"
    llm_api_key: str = "ollama"  # Ollama не проверяет ключ, но клиент требует значение
    llm_temperature: float = 0.1
    llm_max_tokens: int = 1024
    llm_timeout: int = 300  # CPU-инференс медленный — увеличенный таймаут

    # Максимум итераций доработки ответа по замечаниям критика.
    max_revisions: int = 2

    # Идентификатор клиента чата для LTM и сохранённого профиля пассажира.
    agent_user_id: str = "default"

    # --- Память агента (STM / LTM) ---
    memory_enabled: bool = True
    # Режим пакетной оценки: не читать/писать LTM (изоляция прогона, §4.9.2).
    eval_mode: bool = False
    memory_max_turns: int = 10  # порог summarize (число сообщений в истории)
    memory_context_turns: int = 6  # сколько последних реплик в промпт
    memory_top_k: int = 3  # эпизодов из Qdrant
    memory_episodic_collection: str = "agent_episodic"
    memory_episodic_ttl_days: int = 90

    # Checkpoint LangGraph: postgres | memory
    checkpoint_backend: str = "postgres"

    # --- Observability ---
    observability_enabled: bool = True
    log_level: str = "INFO"
    log_format: str = "json"  # json | console
    log_redact_pii: bool = True

    metrics_enabled: bool = True
    metrics_port: int = 9100

    langfuse_enabled: bool = True
    langfuse_host: str = "http://localhost:3000"
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_capture_content: bool = True
    langfuse_debug: bool = False

    # Итоговый JSON на ход (раздел 4.2 / золотой датасет 4.4)
    log_turn_summary: bool = True
    log_include_query: bool = True
    turn_summary_jsonl_path: str = "logs/turn_summaries.jsonl"

    # Метрики инференса Ollama / vLLM
    llm_metrics_enabled: bool = True
    llm_metrics_interval_sec: int = 15
    ollama_metrics_url: str = ""
    vllm_metrics_url: str = ""

    @property
    def pg_url(self) -> str:
        from urllib.parse import quote_plus

        user = quote_plus(self.pg_user)
        password = quote_plus(self.pg_password)
        return f"postgresql://{user}:{password}@{self.pg_host}:{self.pg_port}/{self.pg_database}"

    @property
    def pg_conninfo(self) -> str:
        return (
            f"host={self.pg_host} port={self.pg_port} "
            f"dbname={self.pg_database} user={self.pg_user} "
            f"password={self.pg_password}"
        )


settings = Settings()
