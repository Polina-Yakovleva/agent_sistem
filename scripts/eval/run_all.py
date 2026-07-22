"""Сквозной прогон валидации агента и сборка итогового отчёта."""

from __future__ import annotations

import argparse
import traceback
from typing import Callable, Optional

from scripts.eval import checkpoint as ckpt
from scripts.eval import data_quality
from scripts.eval.loader import Dataset, load_dataset
from scripts.eval.metrics import PassStat, SectionResult
from scripts.eval.report import (
    Block,
    ValidationReport,
    qualitative_block,
    section_from_pass_stat,
    write_report,
)
from scripts.eval.thresholds import Light, ValidationContext, resolve_context

_SECTION_ORDER = ("e2e", "memory", "reflection", "stability", "baseline")


def _limit(cases: list, n: int) -> list:
    return cases if n <= 0 else cases[:n]


def _na(name: str, exc: Exception) -> SectionResult:
    return SectionResult(
        name=name, light=Light.NA, comment=f"не проведён: {type(exc).__name__}: {exc}"
    )


def _want(key: str, from_section: Optional[str], only_section: Optional[str]) -> bool:
    """Нужно ли запускать раздел сейчас."""
    if only_section:
        return key == only_section
    if not from_section:
        return True
    if from_section not in _SECTION_ORDER:
        raise ValueError(f"неизвестный --from-section={from_section!r}")
    return _SECTION_ORDER.index(key) >= _SECTION_ORDER.index(from_section)


def _resolve(
    stem: Optional[str],
    key: str,
    *,
    want: bool,
    force: bool,
    skipped_comment: str,
    compute: Callable[[], SectionResult],
) -> SectionResult:
    """Взять раздел из файла или посчитать и сразу сохранить."""
    if stem and not force:
        cached = ckpt.load_section(stem, key)
        if cached is not None:
            print(f"Раздел [{key}] из чекпоинта ({cached.light.value})")
            return cached
    if not want:
        return SectionResult(name=key, light=Light.NA, comment=skipped_comment)
    section = compute()
    if stem:
        ckpt.save_section(stem, key, section)
    return section


def _e2e_from_checkpoint(
    dataset: Dataset, stem: str, limit: int
) -> tuple[list, dict[str, Optional[bool]], PassStat, str]:
    all_e2e = _limit(dataset.suite("e2e").cases if dataset.suite("e2e") else [], limit)
    path = ckpt.checkpoint_path("e2e", stem)
    rows = {r["id"]: r for r in ckpt.load_case_rows(path) if not r.get("skipped")}
    e2e_by_id: dict[str, Optional[bool]] = {}
    graded_cases = []
    for case in all_e2e:
        row = rows.get(case.id)
        if row is None:
            continue
        e2e_by_id[case.id] = row.get("passed")
        graded_cases.append(case)
    passed = sum(1 for v in e2e_by_id.values() if v is True)
    total = sum(1 for v in e2e_by_id.values() if v is not None)
    return (
        graded_cases,
        e2e_by_id,
        PassStat(passed=passed, total=total),
        f"из чекпоинта {path.name}; n={total}",
    )


def _build_e2e_bundle(
    dataset: Dataset,
    ctx: ValidationContext,
    limit: int,
    *,
    e2e_start: int,
    stem: Optional[str],
    want_e2e: bool,
    force: bool,
) -> tuple[list, dict[str, Optional[bool]], list[SectionResult]]:
    """E2E + инструменты + планирование (прогон или чекпоинт)."""
    from scripts.eval.runner import AgentRunner
    from scripts.eval.suite_eval import evaluate_cases, per_tool_quality

    all_e2e = _limit(dataset.suite("e2e").cases if dataset.suite("e2e") else [], limit)
    e2e_cases = all_e2e
    e2e_by_id: dict[str, Optional[bool]] = {}
    out: list[SectionResult] = []

    # Уже сохранённые секции e2e-блока.
    if stem and not force and ckpt.has_section(stem, "e2e"):
        s_e2e = ckpt.load_section(stem, "e2e")
        s_tools = ckpt.load_section(stem, "tools")
        s_plan = ckpt.load_section(stem, "planning")
        assert s_e2e is not None
        # Если tools/planning ещё не сохранены — восстановить из e2e.jsonl.
        if (s_tools is None or s_plan is None) and ckpt.checkpoint_path("e2e", stem).exists():
            graded, e2e_by_id, stat, comment = _e2e_from_checkpoint(dataset, stem, limit)
            if s_tools is None:
                tool_q = per_tool_quality(graded, e2e_by_id)
                worst_tool = min((v["pass_rate"] for v in tool_q.values() if v["n"]), default=1.0)
                s_tools = SectionResult(
                    name="Инструменты",
                    light=ctx.light_for_pass_rate(worst_tool, 1),
                    comment=f"худший инструмент pass-rate={worst_tool:.1%} (из чекпоинта)",
                    metrics={"per_tool": tool_q},
                )
                ckpt.save_section(stem, "tools", s_tools)
            if s_plan is None:
                plan_vals = [v for v in e2e_by_id.values() if v is not None]
                s_plan = section_from_pass_stat(
                    "Планирование",
                    PassStat(sum(1 for v in plan_vals if v), len(plan_vals)),
                    ctx,
                    comment="агрегировано по кейсам с plan_*_ok (чекпоинт)",
                )
                ckpt.save_section(stem, "planning", s_plan)
            out = [s_e2e, s_tools, s_plan]
            return graded or all_e2e, e2e_by_id, out
        out.append(s_e2e)
        if s_tools:
            out.append(s_tools)
        if s_plan:
            out.append(s_plan)
        if ckpt.checkpoint_path("e2e", stem).exists():
            graded, e2e_by_id, _, _ = _e2e_from_checkpoint(dataset, stem, limit)
            return graded or all_e2e, e2e_by_id, out
        return all_e2e, e2e_by_id, out

    # Пропуск живого e2e: только jsonl.
    if not want_e2e and stem and ckpt.checkpoint_path("e2e", stem).exists():
        graded, e2e_by_id, stat, comment = _e2e_from_checkpoint(dataset, stem, limit)
        print(f"E2E из чекпоинта: {comment}")
        s_e2e = section_from_pass_stat("Качество прогноза end-to-end", stat, ctx, comment=comment)
        tool_q = per_tool_quality(graded, e2e_by_id)
        worst_tool = min((v["pass_rate"] for v in tool_q.values() if v["n"]), default=1.0)
        s_tools = SectionResult(
            name="Инструменты",
            light=ctx.light_for_pass_rate(worst_tool, 1),
            comment=f"худший инструмент pass-rate={worst_tool:.1%} (из чекпоинта)",
            metrics={"per_tool": tool_q},
        )
        plan_vals = [v for v in e2e_by_id.values() if v is not None]
        s_plan = section_from_pass_stat(
            "Планирование",
            PassStat(sum(1 for v in plan_vals if v), len(plan_vals)),
            ctx,
            comment="агрегировано по кейсам с plan_*_ok (чекпоинт)",
        )
        if stem:
            ckpt.save_section(stem, "e2e", s_e2e)
            ckpt.save_section(stem, "tools", s_tools)
            ckpt.save_section(stem, "planning", s_plan)
        return graded or all_e2e, e2e_by_id, [s_e2e, s_tools, s_plan]

    if not want_e2e:
        na = SectionResult(name="Качество прогноза end-to-end", light=Light.NA, comment="пропущено")
        return (
            all_e2e,
            {},
            [
                na,
                _na("Инструменты", RuntimeError("e2e skipped")),
                _na("Планирование", RuntimeError("e2e skipped")),
            ],
        )

    if e2e_start > 0 and stem:
        path = ckpt.checkpoint_path("e2e", stem)
        ckpt.seed_skipped(
            path,
            [c.id for c in e2e_cases[:e2e_start]],
            note=f"пропущены при дозапуске (--e2e-start {e2e_start})",
        )
        e2e_cases = e2e_cases[e2e_start:]
        print(f"Дозапуск e2e: пропуск первых {e2e_start}, осталось {len(e2e_cases)} кейсов")

    try:
        with AgentRunner() as runner:
            e2e = evaluate_cases(runner, e2e_cases, suite="e2e", checkpoint_stem=stem)
        e2e_by_id = {c.case_id: c.passed for c in e2e.cases}
        comment = f"дозапуск с кейса #{e2e_start}; n={e2e.stat.total}" if e2e_start else ""
        s_e2e = section_from_pass_stat(
            "Качество прогноза end-to-end", e2e.stat, ctx, comment=comment
        )
        tool_q = per_tool_quality(e2e_cases, e2e_by_id)
        worst_tool = min((v["pass_rate"] for v in tool_q.values() if v["n"]), default=1.0)
        s_tools = SectionResult(
            name="Инструменты",
            light=ctx.light_for_pass_rate(worst_tool, 1),
            comment=f"худший инструмент pass-rate={worst_tool:.1%}",
            metrics={"per_tool": tool_q},
        )
        plan_vals = [v for v in e2e_by_id.values() if v is not None]
        s_plan = section_from_pass_stat(
            "Планирование",
            PassStat(sum(1 for v in plan_vals if v), len(plan_vals)),
            ctx,
            comment="агрегировано по кейсам с plan_*_ok",
        )
        if stem:
            ckpt.save_section(stem, "e2e", s_e2e)
            ckpt.save_section(stem, "tools", s_tools)
            ckpt.save_section(stem, "planning", s_plan)
        return e2e_cases, e2e_by_id, [s_e2e, s_tools, s_plan]
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc()
        return (
            e2e_cases,
            {},
            [
                _na("Качество прогноза end-to-end", exc),
                _na("Инструменты", exc),
                _na("Планирование", exc),
            ],
        )


def _run_memory(dataset: Dataset, ctx: ValidationContext, limit: int) -> SectionResult:
    from scripts.eval.multiturn import evaluate_multiturn
    from scripts.eval.rag_eval import evaluate_rag
    from scripts.eval.runner import AgentRunner

    mt_cases = _limit(dataset.suite("multiturn").cases if dataset.suite("multiturn") else [], limit)
    with AgentRunner() as runner:
        mt = evaluate_multiturn(runner, mt_cases)
    graded = [m for m in mt if m.passed is not None]
    mt_stat = PassStat(sum(1 for m in graded if m.passed), len(graded))
    rag = evaluate_rag(dataset.suite("rag").cases if dataset.suite("rag") else [])
    return SectionResult(
        name="Память",
        light=ctx.light_for_pass_rate(mt_stat.rate, mt_stat.total),
        comment=f"multiturn pass-rate={mt_stat.rate:.1%}; RAG recall@k={rag.recall_at_k:.2f}",
        metrics={"multiturn": mt_stat.rate, "rag": rag.as_dict()},
    )


def _run_reflection(dataset: Dataset, e2e_cases: list, limit: int) -> SectionResult:
    from scripts.eval.reflection_ablation import (
        run_out_of_scope,
        run_reflection_ablation,
    )
    from scripts.eval.runner import AgentRunner

    refl = run_reflection_ablation(_limit(e2e_cases, min(limit or 10, 10)))
    oos_suite = dataset.suite("out_of_scope")
    boundary_rate = None
    if oos_suite and oos_suite.cases:
        with AgentRunner() as runner:
            oos = run_out_of_scope(runner, _limit(oos_suite.cases, limit))
        boundary_rate = oos.refusal_rate
    light = Light.GREEN if refl.critic_gain >= 0 else Light.YELLOW
    if boundary_rate is not None and boundary_rate < 0.7:
        light = Light.YELLOW if light == Light.GREEN else light
    comment = f"gain критика={refl.critic_gain:+.1%}"
    if boundary_rate is not None:
        comment += f"; корректный отказ вне домена={boundary_rate:.1%}"
    return SectionResult(
        name="Рефлексия",
        light=light,
        comment=comment,
        metrics={"critic": refl.__dict__, "boundary_refusal_rate": boundary_rate},
    )


def _run_stability(e2e_cases: list, limit: int) -> SectionResult:
    from scripts.eval.robustness import run_robustness
    from scripts.eval.runner import AgentRunner

    with AgentRunner() as runner:
        rob = run_robustness(runner, _limit(e2e_cases, min(limit or 10, 10)))
    worst_delta = min((r.delta for r in rob), default=0.0)
    light = (
        Light.GREEN
        if worst_delta >= -0.05
        else (Light.YELLOW if worst_delta >= -0.15 else Light.RED)
    )
    return SectionResult(
        name="Стабильность",
        light=light,
        comment=f"худшее падение pass-rate при аугментации={worst_delta:+.1%}",
        metrics={"per_kind": [r.__dict__ for r in rob]},
    )


def _run_baseline(dataset: Dataset, ctx: ValidationContext, limit: int) -> SectionResult:
    from scripts.eval.baseline import compare_alternative, run_baseline
    from scripts.eval.runner import AgentRunner
    from scripts.eval.suite_eval import evaluate_cases

    cases = _limit(dataset.suite("e2e").cases if dataset.suite("e2e") else [], min(limit or 15, 15))
    with AgentRunner() as runner:
        agent_run = evaluate_cases(runner, cases, suite="e2e")
    agent_rate = agent_run.stat.rate
    base = run_baseline(cases)
    verdict = compare_alternative(agent_rate, base["pass_rate"], ctx)
    return SectionResult(
        name="Baseline vs агент",
        light=verdict.light,
        comment=f"{verdict.verdict}: Δ={verdict.delta_pp:+.1f} п.п. "
        f"(агент {agent_rate:.1%}, baseline {base['pass_rate']:.1%})",
        metrics=verdict.__dict__,
    )


def _quantitative_block(
    dataset: Dataset,
    ctx: ValidationContext,
    limit: int,
    offline: bool,
    *,
    e2e_start: int = 0,
    checkpoint_stem: Optional[str] = None,
    from_section: Optional[str] = None,
    only_section: Optional[str] = None,
    force: bool = False,
) -> Block:
    sections: list[SectionResult] = []
    sections.append(data_quality.to_section(data_quality.analyze(dataset)))

    if offline:
        for name in (
            "Качество прогноза end-to-end",
            "Инструменты",
            "Планирование",
            "Память",
            "Рефлексия",
            "Стабильность",
            "Data drift",
        ):
            sections.append(
                SectionResult(name=name, light=Light.NA, comment="offline-режим: пропущено")
            )
        return Block(name="Количественный анализ", sections=sections)

    stem = checkpoint_stem
    skip = "пропущено (--from-section/--only-section)"

    e2e_cases, _e2e_by_id, e2e_secs = _build_e2e_bundle(
        dataset,
        ctx,
        limit,
        e2e_start=e2e_start,
        stem=stem,
        want_e2e=_want("e2e", from_section, only_section),
        force=force,
    )
    # Починить имена NA-заглушек для памяти/рефлексии
    for s in e2e_secs:
        if s.name == "e2e":
            s.name = "Качество прогноза end-to-end"
    sections.extend(e2e_secs)

    def _fix_name(sec: SectionResult, name: str) -> SectionResult:
        if sec.name in _SECTION_ORDER or sec.name == "e2e":
            sec.name = name
        return sec

    sections.append(
        _fix_name(
            _resolve(
                stem,
                "memory",
                want=_want("memory", from_section, only_section),
                force=force,
                skipped_comment=skip,
                compute=lambda: _run_memory(dataset, ctx, limit),
            ),
            "Память",
        )
    )
    sections.append(
        _fix_name(
            _resolve(
                stem,
                "reflection",
                want=_want("reflection", from_section, only_section),
                force=force,
                skipped_comment=skip,
                compute=lambda: _run_reflection(dataset, e2e_cases, limit),
            ),
            "Рефлексия",
        )
    )
    sections.append(
        _fix_name(
            _resolve(
                stem,
                "stability",
                want=_want("stability", from_section, only_section),
                force=force,
                skipped_comment=skip,
                compute=lambda: _run_stability(e2e_cases, limit),
            ),
            "Стабильность",
        )
    )
    sections.append(
        SectionResult(
            name="Data drift",
            light=Light.NA,
            comment="не проводится: нет валидного OOT-среза из БД "
            "(синтетические «будущие даты» отключены; зона риска)",
        )
    )
    return Block(name="Количественный анализ", sections=sections)


def _alt_block(
    dataset: Dataset,
    ctx: ValidationContext,
    limit: int,
    offline: bool,
    *,
    checkpoint_stem: Optional[str] = None,
    from_section: Optional[str] = None,
    only_section: Optional[str] = None,
    force: bool = False,
) -> Block:
    if offline:
        return Block(
            name="Альтернативное моделирование",
            sections=[SectionResult("Baseline vs агент", Light.NA, "offline-режим: пропущено")],
        )

    def _compute() -> SectionResult:
        try:
            return _run_baseline(dataset, ctx, limit)
        except Exception as exc:  # noqa: BLE001
            return _na("Baseline vs агент", exc)

    sec = _resolve(
        checkpoint_stem,
        "baseline",
        want=_want("baseline", from_section, only_section),
        force=force,
        skipped_comment="пропущено (--from-section/--only-section)",
        compute=_compute,
    )
    if sec.name == "baseline":
        sec.name = "Baseline vs агент"
    return Block(name="Альтернативное моделирование", sections=[sec])


def run(
    significance: Optional[str] = None,
    validation_type: Optional[str] = None,
    limit: int = 0,
    offline: bool = False,
    *,
    e2e_start: int = 0,
    checkpoint_stem: Optional[str] = None,
    from_section: Optional[str] = None,
    only_section: Optional[str] = None,
    force: bool = False,
) -> ValidationReport:
    ctx = resolve_context(significance, validation_type)
    dataset = load_dataset()
    report = ValidationReport(context=ctx)
    report.blocks.append(qualitative_block())
    report.blocks.append(
        _quantitative_block(
            dataset,
            ctx,
            limit,
            offline,
            e2e_start=e2e_start,
            checkpoint_stem=checkpoint_stem,
            from_section=from_section,
            only_section=only_section,
            force=force,
        )
    )
    report.blocks.append(
        _alt_block(
            dataset,
            ctx,
            limit,
            offline,
            checkpoint_stem=checkpoint_stem,
            from_section=from_section,
            only_section=only_section,
            force=force,
        )
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Валидация LLM-агента")
    parser.add_argument("--significance", default=None, help="СЗ модели: A/B/C/D (дефолт C)")
    parser.add_argument("--validation-type", default=None, help="primary | monitoring")
    parser.add_argument(
        "--limit", type=int, default=0, help="ограничить число кейсов на суит (0 — все)"
    )
    parser.add_argument(
        "--offline", action="store_true", help="только offline-разделы (без сервисов)"
    )
    parser.add_argument("--stem", default="validation_report", help="имя файла отчёта")
    parser.add_argument(
        "--e2e-start",
        type=int,
        default=0,
        help="дозапуск e2e с индекса кейса",
    )
    parser.add_argument(
        "--checkpoint",
        action="store_true",
        help="писать/читать чекпоинты (вкл. автоматически при --only/--from-section)",
    )
    parser.add_argument(
        "--from-section",
        choices=list(_SECTION_ORDER),
        default=None,
        help="дозапуск с раздела (готовые section-*.json не пересчитываются)",
    )
    parser.add_argument(
        "--only-section",
        choices=list(_SECTION_ORDER),
        default=None,
        help="прогнать ОДИН раздел и сохранить (рекомендуется при нестабильном туннеле)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="пересчитать раздел даже если есть section-*.json",
    )
    args = parser.parse_args()

    use_ckpt = bool(args.checkpoint or args.e2e_start or args.from_section or args.only_section)
    report = run(
        significance=args.significance,
        validation_type=args.validation_type,
        limit=args.limit,
        offline=args.offline,
        e2e_start=args.e2e_start,
        checkpoint_stem=args.stem if use_ckpt else None,
        from_section=args.from_section,
        only_section=args.only_section,
        force=args.force,
    )
    paths = write_report(report, stem=args.stem)
    print(f"Итоговый светофор: {report.overall_light.value.upper()}")
    print(f"Markdown: {paths['markdown']}")
    print(f"JSON: {paths['json']}")
    if args.only_section:
        print(
            f"Готово: только раздел [{args.only_section}]. "
            "Когда туннель снова жив — следующий --only-section."
        )


if __name__ == "__main__":
    main()
