"""Чекпоинты покейсного прогона и результатов разделов (дозапуск после обрыва)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Optional

from scripts.eval.metrics import SectionResult
from scripts.eval.paths import checkpoints_dir
from scripts.eval.thresholds import Light


def checkpoint_path(suite: str, stem: str = "full_dataset_report") -> Path:
    return checkpoints_dir() / f"{stem}__{suite}.jsonl"


def section_path(section_key: str, stem: str = "full_dataset_report") -> Path:
    return checkpoints_dir() / f"{stem}__section_{section_key}.json"


def load_done_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    done: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        cid = row.get("id")
        if cid:
            done.add(str(cid))
    return done


def append_case(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_case_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def seed_skipped(path: Path, case_ids: Iterable[str], *, note: str) -> None:
    """Пометить кейсы как пропущенные (уже прогнаны до обрыва, без метрик в агрегате)."""
    done = load_done_ids(path)
    for cid in case_ids:
        if cid in done:
            continue
        append_case(
            path,
            {"id": cid, "passed": None, "skipped": True, "note": note, "status": "skipped"},
        )


def save_section(stem: str, section_key: str, section: SectionResult) -> Path:
    """Сохранить результат раздела сразу после завершения (устойчивость к обрыву туннеля)."""
    path = section_path(section_key, stem)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "key": section_key,
        "name": section.name,
        "light": section.light.value if isinstance(section.light, Light) else str(section.light),
        "comment": section.comment,
        "metrics": section.metrics,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved section [{section_key}] -> {path.name}")
    return path


def load_section(stem: str, section_key: str) -> Optional[SectionResult]:
    path = section_path(section_key, stem)
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return SectionResult(
        name=data["name"],
        light=Light(data["light"]),
        comment=data.get("comment", ""),
        metrics=data.get("metrics") or {},
    )


def has_section(stem: str, section_key: str) -> bool:
    return section_path(section_key, stem).exists()
