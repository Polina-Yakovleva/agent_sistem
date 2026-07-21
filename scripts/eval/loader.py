"""Загрузка манифеста и суитов золотого датасета."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml

from scripts.eval.paths import datasets_dir, manifest_path
from scripts.eval.schema import Case


@dataclass
class Suite:
    """Один суит датасета (e2e / rag / booking / multiturn / ...)."""

    name: str
    file: str
    target_count: int
    cases: list[Case]

    @property
    def count(self) -> int:
        return len(self.cases)

    @property
    def count_matches_target(self) -> bool:
        return self.target_count == 0 or self.count == self.target_count


@dataclass
class Dataset:
    """Загруженный датасет: манифест + суиты."""

    version: int
    description: str
    suites: dict[str, Suite]

    def all_cases(self) -> list[Case]:
        out: list[Case] = []
        for suite in self.suites.values():
            out.extend(suite.cases)
        return out

    def suite(self, name: str) -> Optional[Suite]:
        return self.suites.get(name)

    def count_report(self) -> list[dict]:
        """Сверка count vs target_count по каждому суиту (Раздел 1: качество данных)."""
        rows: list[dict] = []
        for name, suite in self.suites.items():
            rows.append(
                {
                    "suite": name,
                    "count": suite.count,
                    "target_count": suite.target_count,
                    "matches": suite.count_matches_target,
                }
            )
        return rows


def _read_yaml(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_suite_file(path: Path, *, name: str, target_count: int = 0) -> Suite:
    data = _read_yaml(path)
    suite_name = data.get("suite", name)
    cases = [Case.from_raw(c, suite=suite_name) for c in (data.get("cases") or [])]
    return Suite(name=name, file=path.name, target_count=target_count, cases=cases)


def load_dataset(base_dir: Optional[Path] = None) -> Dataset:
    """Загрузить весь датасет по манифесту."""
    base = base_dir or datasets_dir()
    manifest_file = (base / "manifest.yaml") if base_dir else manifest_path()
    manifest = _read_yaml(manifest_file)

    suites: dict[str, Suite] = {}
    for split_name, split in (manifest.get("splits") or {}).items():
        file_name = split.get("file")
        target = int(split.get("target_count", 0))
        path = base / file_name
        if not path.exists():
            continue
        suites[split_name] = load_suite_file(path, name=split_name, target_count=target)

    return Dataset(
        version=int(manifest.get("version", 1)),
        description=manifest.get("description", ""),
        suites=suites,
    )
