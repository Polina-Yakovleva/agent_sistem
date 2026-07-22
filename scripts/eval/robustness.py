"""Стабильность к аугментациям запроса """

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Callable

from scripts.eval.runner import AgentRunner
from scripts.eval.schema import Case
from scripts.eval.scorers.deterministic import case_passed, score_case

_FILLERS = ["пожалуйста", "будьте добры", "срочно", "кстати"]


def _swap_adjacent(text: str, rng: random.Random) -> str:
    chars = list(text)
    letters = [i for i, c in enumerate(chars) if c.isalpha()]
    if len(letters) < 2:
        return text
    i = rng.choice(letters[:-1])
    chars[i], chars[i + 1] = chars[i + 1], chars[i]
    return "".join(chars)


def _drop_char(text: str, rng: random.Random) -> str:
    letters = [i for i, c in enumerate(text) if c.isalpha()]
    if not letters:
        return text
    i = rng.choice(letters)
    return text[:i] + text[i + 1 :]


def typo_augment(text: str, seed: int = 0) -> str:
    """Символьная аугментация: 1-2 опечатки (перестановка/пропуск буквы)."""
    rng = random.Random(seed)
    out = _swap_adjacent(text, rng)
    if len(text) > 12:
        out = _drop_char(out, rng)
    return out


def case_augment(text: str, seed: int = 0) -> str:
    """Регистровая аугментация: рандомизация регистра букв."""
    rng = random.Random(seed)
    return "".join(c.upper() if rng.random() < 0.4 else c.lower() for c in text)


def spacing_augment(text: str, seed: int = 0) -> str:
    """Пробельная аугментация: сдвоенные пробелы и обрезка регистра."""
    rng = random.Random(seed)
    words = text.split()
    return (" " if rng.random() < 0.5 else "  ").join(words).strip()


def word_augment(text: str, seed: int = 0) -> str:
    """Словесная аугментация: вставка шумового слова + локальная перестановка."""
    rng = random.Random(seed)
    words = text.split()
    if len(words) >= 2:
        i = rng.randrange(len(words) - 1)
        words[i], words[i + 1] = words[i + 1], words[i]
    words.insert(0, rng.choice(_FILLERS))
    return " ".join(words)


AUGMENTERS: dict[str, Callable[[str, int], str]] = {
    "typo": typo_augment,
    "case": case_augment,
    "spacing": spacing_augment,
    "word": word_augment,
}


@dataclass
class RobustnessResult:
    kind: str
    base_pass_rate: float
    aug_pass_rate: float
    n: int
    per_case: list[dict] = field(default_factory=list)

    @property
    def delta(self) -> float:
        return self.aug_pass_rate - self.base_pass_rate


def _pass(runner: AgentRunner, case: Case, query: str) -> bool | None:
    outcome = runner.run_turn(query, case_id=case.id)
    return case_passed(score_case(case, outcome))


def run_robustness(
    runner: AgentRunner,
    cases: list[Case],
    kinds: list[str] | None = None,
    seed: int = 13,
) -> list[RobustnessResult]:
    """Сравнить pass-rate на исходных и аугментированных запросах."""
    kinds = kinds or list(AUGMENTERS)
    results: list[RobustnessResult] = []
    for kind in kinds:
        aug = AUGMENTERS[kind]
        base_hits = aug_hits = graded = 0
        per_case: list[dict] = []
        for idx, case in enumerate(cases):
            base = _pass(runner, case, case.user_query)
            perturbed = _pass(runner, case, aug(case.user_query, seed + idx))
            if base is None or perturbed is None:
                continue
            graded += 1
            base_hits += int(base)
            aug_hits += int(perturbed)
            per_case.append({"id": case.id, "base": base, "aug": perturbed})
        n = graded or 1
        results.append(
            RobustnessResult(
                kind=kind,
                base_pass_rate=base_hits / n,
                aug_pass_rate=aug_hits / n,
                n=graded,
                per_case=per_case,
            )
        )
    return results
