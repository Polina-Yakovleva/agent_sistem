"""Статистика pass-rate и свёртка светофоров."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from scripts.eval.thresholds import Light, worst


@dataclass
class PassStat:
    passed: int
    total: int

    @property
    def rate(self) -> float:
        return self.passed / self.total if self.total else 0.0

    def wilson_ci(self, z: float = 1.96) -> tuple[float, float]:
        """Доверительный интервал Уилсона для доли (95% по умолчанию)."""
        n = self.total
        if n == 0:
            return (0.0, 0.0)
        p = self.rate
        denom = 1 + z * z / n
        center = (p + z * z / (2 * n)) / denom
        half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
        return (max(0.0, center - half), min(1.0, center + half))


@dataclass
class SectionResult:
    """Раздел количественного/качественного анализа."""

    name: str
    light: Light
    comment: str = ""
    metrics: dict = field(default_factory=dict)
    subtests: list["SectionResult"] = field(default_factory=list)


def rollup(sections: list[SectionResult]) -> Light:
    """Светофор блока = наихудший среди разделов."""
    return worst([s.light for s in sections])
