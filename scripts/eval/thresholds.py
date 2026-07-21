from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Light(str, Enum):
    """Светофор результата теста/раздела/модели."""

    GREEN = "green"
    YELLOW = "yellow"
    RED = "red"
    NA = "na"  # тест не применим или не проводился


# Порядок «строгости» для свёртки «наихудший из…» (NA не участвует).
_SEVERITY = {Light.GREEN: 0, Light.YELLOW: 1, Light.RED: 2}


def worst(lights: list[Light]) -> Light:
    """Свёртка «наихудший среди тестов» (неприменимые/непроведённые игнорируются)."""
    applicable = [x for x in lights if x in _SEVERITY]
    if not applicable:
        return Light.NA
    return max(applicable, key=lambda x: _SEVERITY[x])


class SignificanceLevel(str, Enum):
    A = "A"
    B = "B"
    C = "C"
    D = "D"


class ValidationType(str, Enum):
    PRIMARY = "primary"  # первичная — перед пилотом/промом
    MONITORING = "monitoring"  # плановый / событийный мониторинг


@dataclass(frozen=True)
class SignificanceProfile:
    """Пороги и требования артефактов для конкретной СЗ."""

    level: SignificanceLevel
    # Минимальное статзначимое улучшение метрики на OOS/OOT (альтмоделирование), п.п.
    alt_improvement_pp: float
    # Требуется ли разметка milestones и эталонных ответов.
    milestones_required: bool
    # Пороги итоговой pass-rate для светофора раздела (доля пройденных кейсов).
    green_min: float
    yellow_min: float


# СЗ A/B — жёсткие пороги (улучшение ≥1 п.п., нужны milestones);
# СЗ C/D — мягче (улучшение ≥3 п.п., milestones не обязательны).
SIGNIFICANCE_PROFILES: dict[SignificanceLevel, SignificanceProfile] = {
    SignificanceLevel.A: SignificanceProfile(
        SignificanceLevel.A,
        alt_improvement_pp=1.0,
        milestones_required=True,
        green_min=0.90,
        yellow_min=0.80,
    ),
    SignificanceLevel.B: SignificanceProfile(
        SignificanceLevel.B,
        alt_improvement_pp=1.0,
        milestones_required=True,
        green_min=0.90,
        yellow_min=0.80,
    ),
    SignificanceLevel.C: SignificanceProfile(
        SignificanceLevel.C,
        alt_improvement_pp=3.0,
        milestones_required=False,
        green_min=0.85,
        yellow_min=0.70,
    ),
    SignificanceLevel.D: SignificanceProfile(
        SignificanceLevel.D,
        alt_improvement_pp=3.0,
        milestones_required=False,
        green_min=0.80,
        yellow_min=0.65,
    ),
}


@dataclass
class ValidationContext:
    """Зафиксированные на Шаге 0 параметры валидации."""

    significance: SignificanceLevel = SignificanceLevel.C
    validation_type: ValidationType = ValidationType.PRIMARY
    # Подтверждение статуса «агент» (три обязательных аспекта).
    is_planning: bool = True  # недетерминированный план (planner_node + LLM)
    is_autonomous: bool = True  # без подтверждения каждого шага (кроме HITL booking)
    is_tool_using: bool = True  # самостоятельный выбор инструментов (ReAct-субагенты)
    notes: list[str] = field(default_factory=list)

    @property
    def is_agent(self) -> bool:
        return self.is_planning and self.is_autonomous and self.is_tool_using

    @property
    def profile(self) -> SignificanceProfile:
        return SIGNIFICANCE_PROFILES[self.significance]

    def light_for_pass_rate(self, pass_rate: float, n: int) -> Light:
        """Светофор раздела по доле пройденных кейсов относительно порогов СЗ."""
        if n == 0:
            return Light.NA
        p = self.profile
        if pass_rate >= p.green_min:
            return Light.GREEN
        if pass_rate >= p.yellow_min:
            return Light.YELLOW
        return Light.RED

    def summary(self) -> dict:
        return {
            "is_agent": self.is_agent,
            "aspects": {
                "planning": self.is_planning,
                "autonomy": self.is_autonomous,
                "tool_using": self.is_tool_using,
            },
            "significance": self.significance.value,
            "validation_type": self.validation_type.value,
            "alt_improvement_pp": self.profile.alt_improvement_pp,
            "milestones_required": self.profile.milestones_required,
            "pass_rate_thresholds": {
                "green_min": self.profile.green_min,
                "yellow_min": self.profile.yellow_min,
            },
            "notes": self.notes,
        }


def resolve_context(
    significance: str | None = None,
    validation_type: str | None = None,
) -> ValidationContext:
    """Собрать контекст валидации из строковых параметров (CLI/env)."""
    ctx = ValidationContext()
    if significance:
        ctx.significance = SignificanceLevel(significance.strip().upper())
    if validation_type:
        ctx.validation_type = ValidationType(validation_type.strip().lower())
    ctx.notes.append(
        "Статус «агент» подтверждён архитектурой graph.py: planner_node (Planning), "
        "ReAct-субагенты со свободным выбором инструментов (Tool using), автономный "
        "прогон без подтверждения каждого шага, кроме HITL в booking (Autonomy)."
    )
    if ctx.significance in (SignificanceLevel.A, SignificanceLevel.B):
        ctx.notes.append(
            "СЗ A/B: требуются разметка milestones и эталонные ответы; порог улучшения "
            "при альтмоделировании ≥1 п.п."
        )
    else:
        ctx.notes.append(
            "СЗ C/D (дефолт C): milestones/эталоны не обязательны; порог улучшения "
            "при альтмоделировании ≥3 п.п. Эскалировать до B при пром-бронированиях/PII."
        )
    return ctx
