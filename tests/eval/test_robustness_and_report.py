"""Тесты аугментаций стабильности и рендеринга отчёта (offline)."""

from scripts.eval.metrics import PassStat, SectionResult
from scripts.eval.report import (
    Block,
    ValidationReport,
    render_markdown,
    section_from_pass_stat,
)
from scripts.eval.robustness import AUGMENTERS, typo_augment
from scripts.eval.thresholds import Light, ValidationContext


def test_augmenters_deterministic_and_change_text():
    q = "рейсы Москва Стамбул на завтра"
    for name, fn in AUGMENTERS.items():
        a1 = fn(q, 5)
        a2 = fn(q, 5)
        assert a1 == a2, f"{name} not deterministic"


def test_typo_augment_preserves_most_chars():
    q = "нужна ли виза в Турцию"
    aug = typo_augment(q, 1)
    assert aug != q
    assert abs(len(aug) - len(q)) <= 2


def test_section_from_pass_stat_light():
    ctx = ValidationContext()
    sec = section_from_pass_stat("e2e", PassStat(9, 10), ctx)
    assert sec.light == Light.GREEN
    assert sec.metrics["pass_rate"] == 0.9


def test_report_rollup_and_markdown():
    ctx = ValidationContext()
    quant = Block(
        name="Количественный анализ",
        sections=[
            SectionResult("A", Light.GREEN),
            SectionResult("B", Light.RED),
            SectionResult("C", Light.NA),
        ],
    )
    report = ValidationReport(context=ctx, blocks=[quant])
    assert quant.light == Light.RED
    assert report.overall_light == Light.RED
    md = render_markdown(report)
    assert "Итоговый светофор" in md
    assert "Количественный анализ" in md
