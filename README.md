# Agent Runtime Package

Минимальная production-oriented упаковка агента для последующей контейнеризации и API-слоя.

## Структура проекта

```text
agent_runtime/
├── app/                      # runtime агента
│   ├── agents/               # оркестратор, субагенты, critic, guardrails
│   ├── tools/                # flight / booking (HITL) / compliance / external
│   ├── memory/               # STM (checkpoint) + LTM (эпизоды, профиль)
│   ├── observability/        # логи, метрики, Langfuse
│   ├── api.py                # FastAPI: /v1/chat, /v1/chat/resume, health
│   ├── service.py            # общий диалоговый слой (CLI и API)
│   ├── main.py               # CLI-вход
│   ├── rag.py                # retrieval в Qdrant (compliance)
│   ├── runtime.py            # сборка графа / сессии
│   ├── llm.py                # OpenAI-compatible LLM
│   ├── db.py                 # Postgres
│   ├── config.py             # настройки из окружения
│   └── external_api.py       # погода / отели и др. внешние API
├── examples/
│   └── poc_hitl.py           # PoC: happy path + эскалация HITL
├── tests/                    # unit-тесты + test_poc_hitl
├── .github/workflows/        # ci.yml (ruff + pytest)
├── checkenv.py               # дефолты и проверка .env
├── Dockerfile
├── docker-compose.yml        # api + Postgres + Qdrant
├── requirements.txt
└── requirements-dev.txt
```

`.env` с секретами в git не хранится.

## Быстрый запуск (локально)

1. Установить зависимости:

   ```bash
   pip install -r requirements.txt
   ```

2. Подготовить `.env`:

   ```bash
   python checkenv.py --init
   ```

   Заполнить секреты (`LLM_API_KEY`, `PG_PASSWORD`, …), затем:

   ```bash
   python checkenv.py
   ```

3. CLI:

   ```bash
   python -m app.main
   python -m app.main "Найди рейсы из Москвы в Стамбул"
   ```

4. HTTP API:

   ```bash
   uvicorn app.api:app --host 0.0.0.0 --port 8000
   ```

### API endpoints

| Метод | Путь | Описание |
|-------|------|----------|
| GET | `/health` | liveness |
| GET | `/ready` | readiness (LLM доступен) |
| POST | `/v1/chat` | ход диалога |
| POST | `/v1/chat/resume` | подтверждение HITL |

Пример:

```bash
curl -X POST http://localhost:8000/v1/chat \
  -H "Content-Type: application/json" \
  -d "{\"message\":\"Найди рейсы из Москвы в Стамбул\"}"
```

При `status: interrupted` повторите запрос на `/v1/chat/resume` с тем же `session_id`:

```bash
curl -X POST http://localhost:8000/v1/chat/resume \
  -H "Content-Type: application/json" \
  -d "{\"session_id\":\"...\",\"decision\":\"да\"}"
```

## Docker / Compose

Нужны заполненный `.env` и LLM на хосте.

```bash
docker compose up --build -d
curl http://localhost:8000/health
```

Сервисы: `api` (порт 8000), `postgres` (5432), `qdrant` (6333).  
В контейнере `PG_HOST=postgres`, `QDRANT_HOST=qdrant`, LLM по умолчанию через `host.docker.internal`.

CLI внутри compose:

```bash
docker compose run --rm api python -m app.main "привет"
```

Схема БД и коллекция Qdrant в compose не создаются автоматически — их нужно подготовить заранее.

## Тесты / CI

Unit-тесты (`tests/`) покрывают чистую логику агентов, инструментов и конфига —
LLM, Postgres и Qdrant в них мокаются (`monkeypatch`), реальные сервисы не нужны.

Установить dev-зависимости и запустить тесты локально:

```bash
pip install -r requirements-dev.txt
ruff check .
pytest --cov=app --cov-report=term-missing
```

На каждый push/PR в `main` GitHub Actions (`.github/workflows/ci.yml`) прогоняет
линтер (`ruff`) и тесты (`pytest`).

## PoC: happy path + рискованный путь с эскалацией на человека

`examples/poc_hitl.py` демонстрирует два режима работы на реальном коде инструментов:

1. **Happy path** — безопасная read-only операция (поиск рейсов, `get_flights`)
   выполняется агентом автономно, без участия человека.
2. **Рискованный путь** — необратимая операция (отмена брони, `cancel_reservation`)
   эскалируется на человека через Human-in-the-loop: инструмент вызывает
   `interrupt()`, LangGraph приостанавливает граф и ждёт явного «да/нет».
   Запись в БД происходит только после подтверждения; при «нет» операция отменяется.

Запуск без внешней инфраструктуры (БД мокается in-memory):

```bash
python -m examples.poc_hitl
```

Те же сценарии проверяются в CI: `tests/test_poc_hitl.py` (LLM/Postgres мокаются).

## Готовность к API/Docker

- Точка входа API: `uvicorn app.api:app`.
- Точка входа CLI: `python -m app.main`.
- Общий слой диалога: `app.service.AgentService`.
