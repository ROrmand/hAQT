"""Parse model generations and score against derived CVE gold labels."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable

from src.tasks import BenchmarkExample, TaskName

_CWE_RE = re.compile(r"CWE-\d+", re.IGNORECASE)
_VERSION_TOKEN_RE = re.compile(
    r"\bv?\d+(?:\.\d+){1,4}(?:[-_][A-Za-z0-9]+)?\b",
    re.IGNORECASE,
)


@dataclass
class ExampleScore:
    cve_id: str
    task: str
    correct: bool
    predicted: Any
    gold: Any
    detail: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TaskReport:
    task: str
    n: int
    accuracy: float | None = None
    precision: float | None = None
    recall: float | None = None
    f1: float | None = None
    examples: list[ExampleScore] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        return payload

    def summary(self) -> dict[str, float]:
        out: dict[str, float] = {"n": float(self.n)}
        for key in ("accuracy", "precision", "recall", "f1"):
            value = getattr(self, key)
            if value is not None:
                out[key] = value
        return out


def parse_cwe_prediction(text: str) -> str | None:
    text = (text or "").strip()
    if not text or text.upper().startswith("UNKNOWN"):
        return None
    match = _CWE_RE.search(text)
    return match.group(0).upper() if match else None


def parse_version_prediction(text: str) -> tuple[str, ...]:
    text = (text or "").strip()
    if not text or text.upper().startswith("NONE"):
        return ()
    found = [_normalize_version(m.group(0)) for m in _VERSION_TOKEN_RE.finditer(text)]
    return tuple(dict.fromkeys(found))


def _normalize_version(value: str) -> str:
    return value.lstrip("vV").strip()


def _f1(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def score_cwe_example(example: BenchmarkExample, prediction_text: str) -> ExampleScore:
    predicted = parse_cwe_prediction(prediction_text)
    gold = example.gold_cwe
    gold_set = {c.upper() for c in example.gold_cwes} if example.gold_cwes else set()
    if gold:
        gold_set.add(gold.upper())

    exact = predicted is not None and gold is not None and predicted == gold.upper()
    in_set = predicted is not None and predicted in gold_set
    return ExampleScore(
        cve_id=example.cve_id,
        task=TaskName.CWE_CLASSIFICATION.value,
        correct=exact,
        predicted=predicted,
        gold=gold,
        detail={
            "exact_match": 1.0 if exact else 0.0,
            "in_gold_set": 1.0 if in_set else 0.0,
        },
    )


def score_version_example(example: BenchmarkExample, prediction_text: str) -> ExampleScore:
    predicted = set(parse_version_prediction(prediction_text))
    gold = {_normalize_version(v) for v in example.gold_versions}
    tp = len(predicted & gold)
    precision = tp / len(predicted) if predicted else (1.0 if not gold else 0.0)
    recall = tp / len(gold) if gold else (1.0 if not predicted else 0.0)
    f1 = _f1(precision, recall)
    # "correct" = perfect set match for accuracy-style aggregate
    correct = predicted == gold
    return ExampleScore(
        cve_id=example.cve_id,
        task=TaskName.VERSION_PARSING.value,
        correct=correct,
        predicted=sorted(predicted),
        gold=sorted(gold),
        detail={"precision": precision, "recall": recall, "f1": f1},
    )


def score_example(example: BenchmarkExample, prediction_text: str) -> ExampleScore:
    if example.task == TaskName.CWE_CLASSIFICATION:
        return score_cwe_example(example, prediction_text)
    if example.task == TaskName.VERSION_PARSING:
        return score_version_example(example, prediction_text)
    raise ValueError(f"Unknown task: {example.task}")


def aggregate_cwe(scores: list[ExampleScore]) -> TaskReport:
    n = len(scores)
    acc = sum(1 for s in scores if s.correct) / n if n else None
    soft = sum(s.detail.get("in_gold_set", 0.0) for s in scores) / n if n else None
    return TaskReport(
        task=TaskName.CWE_CLASSIFICATION.value,
        n=n,
        accuracy=acc,
        # Reuse precision slot for soft "predicted CWE in gold set" rate
        precision=soft,
        examples=scores,
    )


def aggregate_version(scores: list[ExampleScore]) -> TaskReport:
    n = len(scores)
    if not n:
        return TaskReport(task=TaskName.VERSION_PARSING.value, n=0, examples=scores)
    acc = sum(1 for s in scores if s.correct) / n
    precision = sum(s.detail["precision"] for s in scores) / n
    recall = sum(s.detail["recall"] for s in scores) / n
    f1 = sum(s.detail["f1"] for s in scores) / n
    return TaskReport(
        task=TaskName.VERSION_PARSING.value,
        n=n,
        accuracy=acc,
        precision=precision,
        recall=recall,
        f1=f1,
        examples=scores,
    )


def score_predictions(
    examples: Iterable[BenchmarkExample],
    predictions: Iterable[str],
) -> dict[str, TaskReport]:
    """Score aligned example/prediction pairs; split reports by task."""
    paired = list(zip(examples, predictions, strict=True))
    cwe_scores: list[ExampleScore] = []
    version_scores: list[ExampleScore] = []
    for example, pred in paired:
        scored = score_example(example, pred)
        if example.task == TaskName.CWE_CLASSIFICATION:
            cwe_scores.append(scored)
        else:
            version_scores.append(scored)

    reports: dict[str, TaskReport] = {}
    if cwe_scores:
        reports[TaskName.CWE_CLASSIFICATION.value] = aggregate_cwe(cwe_scores)
    if version_scores:
        reports[TaskName.VERSION_PARSING.value] = aggregate_version(version_scores)
    return reports


def reports_to_flat_metrics(reports: dict[str, TaskReport]) -> dict[str, float]:
    """Flatten for ProfileMetrics.task_scores / CSV rows."""
    flat: dict[str, float] = {}
    for task, report in reports.items():
        for key, value in report.summary().items():
            flat[f"{task}.{key}"] = value
    return flat
