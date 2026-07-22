"""Регенерация и аугментация золотого датасета.

Отвечает за:
- сверку и синхронизацию ``count``/``target_count`` 
- генерацию производных сплитов покрытия:
  
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from scripts.eval.paths import datasets_dir

# --- Производные сплиты (детерминированно генерируются) -------------------- #
_OUT_OF_SCOPE_QUERIES = [
    "какой сегодня курс доллара к рублю",
    "расскажи анекдот про программистов",
    "как приготовить борщ",
    "реши уравнение 2x + 3 = 7",
    "переведи слово hello на французский",
    "asdf qwerty 12345 zxcv",
    "напиши короткое стихотворение про осень",
    "кто выиграл последний чемпионат мира по футболу",
]

def build_out_of_scope() -> dict:
    cases = []
    for i, q in enumerate(_OUT_OF_SCOPE_QUERIES, 1):
        cases.append(
            {
                "id": f"oos_scope_{i:03d}",
                "suite": "out_of_scope",
                "user_query": q,
                "tags": ["out_of_scope", "boundary"],
                "difficulty": "medium",
                "expect_refusal": True,
                "success_criteria": ["has_final_answer"],
            }
        )
    return {
        "version": 1,
        "suite": "out_of_scope",
        "description": "Запросы вне домена/«мусорные» — ожидается отказ или уточнение",
        "count": len(cases),
        "cases": cases,
    }


def _dump_yaml(path: Path, data: dict) -> None:
    path.write_text(
        yaml.safe_dump(data, allow_unicode=True, sort_keys=False, width=100),
        encoding="utf-8",
    )


def _actual_count(base: Path, file_name: str) -> int:
    path = base / file_name
    if not path.exists():
        return 0
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return len(data.get("cases") or [])


def sync_manifest(base: Path, *, derived: dict[str, str], write: bool) -> list[dict]:
    """Синхронизировать target_count с фактическим и зарегистрировать derived-сплиты."""
    manifest_path = base / "manifest.yaml"
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
    splits = manifest.setdefault("splits", {})

    # Синтетический OOT больше не поддерживается.
    splits.pop("oot", None)

    for name, file_name in derived.items():
        splits.setdefault(name, {"file": file_name})
        splits[name]["file"] = file_name

    report: list[dict] = []
    for name, split in splits.items():
        actual = _actual_count(base, split.get("file", ""))
        old = split.get("target_count")
        report.append({"suite": name, "old_target": old, "actual": actual})
        split["target_count"] = actual

    if write:
        _dump_yaml(manifest_path, manifest)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Регенерация/аугментация датасета")
    parser.add_argument("--check", action="store_true", help="только отчёт, без записи")
    args = parser.parse_args()

    base = datasets_dir()
    write = not args.check

    derived = {
        "out_of_scope": "out_of_scope_cases.yaml",
    }
    if write:
        _dump_yaml(base / "out_of_scope_cases.yaml", build_out_of_scope())
        stale_oot = base / "oot_cases.yaml"
        if stale_oot.exists():
            stale_oot.unlink()

    report = sync_manifest(base, derived=derived, write=write)
    print("Синхронизация манифеста (suite: old_target -> actual):")
    for row in report:
        print(f"  {row['suite']}: {row['old_target']} -> {row['actual']}")
    if args.check:
        print("(режим --check: файлы не записаны)")
    else:
        print(f"Записаны derived-сплиты и манифест в {base}")


if __name__ == "__main__":
    main()