"""Свёртка светофоров и оформление отчёта о валидации ."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from scripts.eval.metrics import PassStat, SectionResult
from scripts.eval.paths import reports_dir
from scripts.eval.thresholds import Light, ValidationContext, worst

_LIGHT_ICON = {
    Light.GREEN: "🟢",
    Light.YELLOW: "🟡",
    Light.RED: "🔴",
    Light.NA: "⚪",
}


def section_from_pass_stat(
    name: str, stat: PassStat, ctx: ValidationContext, comment: str = "", **metrics
) -> SectionResult:
    """Собрать раздел из доли пройденных кейсов + светофор по порогам СЗ."""
    light = ctx.light_for_pass_rate(stat.rate, stat.total)
    lo, hi = stat.wilson_ci()
    base = {
        "n": stat.total,
        "passed": stat.passed,
        "pass_rate": round(stat.rate, 4),
        "wilson_ci95": [round(lo, 4), round(hi, 4)],
    }
    base.update(metrics)
    text = comment or f"pass-rate {stat.rate:.1%} на {stat.total} кейсах"
    return SectionResult(name=name, light=light, comment=text, metrics=base)


@dataclass
class Block:
    """Блок анализа (качественный / количественный / альтмоделирование)."""

    name: str
    sections: list[SectionResult] = field(default_factory=list)

    @property
    def light(self) -> Light:
        return worst([s.light for s in self.sections])


@dataclass
class ValidationReport:
    context: ValidationContext
    blocks: list[Block] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @property
    def overall_light(self) -> Light:
        return worst([b.light for b in self.blocks])

    def not_run(self) -> list[str]:
        """Непроведённые/неприменимые разделы — зона риска."""
        out: list[str] = []
        for block in self.blocks:
            for s in block.sections:
                if s.light == Light.NA:
                    out.append(f"{block.name} / {s.name}: {s.comment}")
        return out

    def as_dict(self) -> dict:
        return {
            "created_at": self.created_at,
            "context": self.context.summary(),
            "overall_light": self.overall_light.value,
            "blocks": [
                {
                    "name": b.name,
                    "light": b.light.value,
                    "sections": [
                        {
                            "name": s.name,
                            "light": s.light.value,
                            "comment": s.comment,
                            "metrics": s.metrics,
                        }
                        for s in b.sections
                    ],
                }
                for b in self.blocks
            ],
            "not_run": self.not_run(),
        }


def render_markdown(report: ValidationReport) -> str:
    ctx = report.context
    lines: list[str] = []
    lines.append("# Отчёт о валидации LLM-агента")
    lines.append("")
    lines.append(f"- Дата: {report.created_at}")
    lines.append(f"- Статус «агент»: {'да' if ctx.is_agent else 'нет'}")
    lines.append(f"- Степень значимости (СЗ): {ctx.significance.value}")
    lines.append(f"- Тип валидации: {ctx.validation_type.value}")
    lines.append(
        f"- **Итоговый светофор: {_LIGHT_ICON[report.overall_light]} "
        f"{report.overall_light.value.upper()}**"
    )
    lines.append("")
    for block in report.blocks:
        lines.append(f"## {block.name} — {_LIGHT_ICON[block.light]}")
        lines.append("")
        lines.append("| Раздел | Светофор | Комментарий |")
        lines.append("|---|---|---|")
        for s in block.sections:
            comment = s.comment.replace("|", "\\|")
            lines.append(f"| {s.name} | {_LIGHT_ICON[s.light]} | {comment} |")
        lines.append("")
    not_run = report.not_run()
    if not_run:
        lines.append("## Не проводились / неприменимо (зона риска)")
        lines.append("")
        for item in not_run:
            lines.append(f"- {item}")
        lines.append("")
    if ctx.notes:
        lines.append("## Примечания Шага 0")
        lines.append("")
        for note in ctx.notes:
            lines.append(f"- {note}")
        lines.append("")
    return "\n".join(lines)


def write_report(report: ValidationReport, *, stem: str = "validation_report") -> dict[str, Path]:
    """Записать отчёт в Markdown и JSON, вернуть пути."""
    out_dir = reports_dir()
    md_path = out_dir / f"{stem}.md"
    json_path = out_dir / f"{stem}.json"
    md_path.write_text(render_markdown(report), encoding="utf-8")
    json_path.write_text(
        json.dumps(report.as_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return {"markdown": md_path, "json": json_path}


def qualitative_block() -> Block:
    """Качественный анализ (Шаг 4) — чек-лист для ручного заполнения валидатором.

    Автоматизировать нельзя: помечаем NA с указанием, что проверяется по
    сопроводительной документации разработчика.
    """
    items = [
        (
            "Качество документации",
            "Проверить по чек-листу Приложения 6 (задачи, архитектура, данные, ПО, отчёт).",
        ),
        (
            "Подход к моделированию",
            "Обоснование LLM vs простых NLP; альтернативы промпта; отсутствие OOS в подборе.",
        ),
        (
            "Дизайн разметки",
            "Auto-only: ручной разметки нет; honeypot/попарное сравнение людьми не проводятся.",
        ),
    ]
    return Block(
        name="Качественный анализ",
        sections=[SectionResult(name=n, light=Light.NA, comment=c) for n, c in items],
    )
