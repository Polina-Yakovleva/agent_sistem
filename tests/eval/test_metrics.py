"""Тесты статистических помощников и свёртки светофоров."""

from scripts.eval.metrics import PassStat
from scripts.eval.thresholds import (
    Light,
    ValidationContext,
    resolve_context,
    worst,
)


def test_pass_stat_rate_and_ci():
    s = PassStat(passed=8, total=10)
    assert s.rate == 0.8
    lo, hi = s.wilson_ci()
    assert 0.0 <= lo < s.rate < hi <= 1.0


def test_worst_rollup_ignores_na():
    assert worst([Light.GREEN, Light.NA]) == Light.GREEN
    assert worst([Light.GREEN, Light.YELLOW]) == Light.YELLOW
    assert worst([Light.YELLOW, Light.RED]) == Light.RED
    assert worst([Light.NA, Light.NA]) == Light.NA


def test_context_thresholds_by_significance():
    ab = resolve_context("A")
    cd = resolve_context("C")
    assert ab.profile.alt_improvement_pp == 1.0
    assert cd.profile.alt_improvement_pp == 3.0
    assert ab.profile.milestones_required is True
    assert cd.profile.milestones_required is False


def test_light_for_pass_rate():
    ctx = ValidationContext()  # СЗ C: green>=0.85, yellow>=0.70
    assert ctx.light_for_pass_rate(0.9, 10) == Light.GREEN
    assert ctx.light_for_pass_rate(0.75, 10) == Light.YELLOW
    assert ctx.light_for_pass_rate(0.5, 10) == Light.RED
    assert ctx.light_for_pass_rate(0.0, 0) == Light.NA
