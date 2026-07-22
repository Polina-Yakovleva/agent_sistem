# Валидация LLM-агента

Харнесс: золотой датасет (`datasets/`) → прогон агента → светофорный отчёт.
Оценка автоматическая (детерминированные критерии + retrieval-метрики + абляции).

Витрина: корневой `QUALITY.md`. Снимок: `eval_reports/full_dataset_report.*`.

## Быстрый старт

```bash
pip install -r requirements-eval.txt -r requirements-dev.txt

python -m scripts.eval.run_all --offline          # без LLM/БД/Qdrant
python -m scripts.eval.run_all --significance C --limit 0
python -m scripts.eval.generate_datasets --check
python -m pytest tests/eval -q
```

## Разделы

| Раздел | Модуль |
|---|---|
| 1. Качество данных | `data_quality.py` (offline) |
| 2. End-to-end | `suite_eval.py`, `scorers/` |
| 3. Инструменты | `per_tool_quality` в `suite_eval.py` |
| 4. Планирование | агрегация по e2e (`plan_*_ok`) |
| 5. Память / RAG | `multiturn.py`, `rag_eval.py` |
| 6. Рефлексия | `reflection_ablation.py` |
| 7. Стабильность | `robustness.py` |
| Baseline | `baseline.py` |

