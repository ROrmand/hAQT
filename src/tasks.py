"""Benchmark task prompts for CVE CWE classification and version parsing."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable

from src.data_loader import CVERecord


class TaskName(str, Enum):
    CWE_CLASSIFICATION = "cwe_classification"
    VERSION_PARSING = "version_parsing"


@dataclass(frozen=True)
class BenchmarkExample:
    """One promptable example with gold labels for scoring."""

    task: TaskName
    cve_id: str
    prompt: str
    gold_cwe: str | None = None
    gold_cwes: tuple[str, ...] = ()
    gold_versions: tuple[str, ...] = ()


_CWE_SYSTEM = (
    "You are a vulnerability analyst. Read the CVE description and identify the "
    "primary CWE weakness ID. Reply with ONLY a CWE id like CWE-79. "
    "If unknown, reply UNKNOWN."
)

_VERSION_SYSTEM = (
    "You are a vulnerability analyst. Extract software version numbers mentioned "
    "as affected, fixed, or bounded in the CVE text (e.g. 1.4.0, 3.2.1). "
    "Reply with a comma-separated list of versions only. If none, reply NONE."
)


def build_cwe_prompt(record: CVERecord) -> str:
    return (
        f"{_CWE_SYSTEM}\n\n"
        f"CVE: {record.cve_id}\n"
        f"Description:\n{record.description}\n\n"
        "Primary CWE:"
    )


def build_version_prompt(record: CVERecord) -> str:
    return (
        f"{_VERSION_SYSTEM}\n\n"
        f"CVE: {record.cve_id}\n"
        f"Description:\n{record.description}\n\n"
        "Versions:"
    )


def build_cwe_examples(
    records: Iterable[CVERecord],
    *,
    require_cwe: bool = True,
) -> list[BenchmarkExample]:
    examples: list[BenchmarkExample] = []
    for record in records:
        if require_cwe and not record.primary_cwe:
            continue
        examples.append(
            BenchmarkExample(
                task=TaskName.CWE_CLASSIFICATION,
                cve_id=record.cve_id,
                prompt=build_cwe_prompt(record),
                gold_cwe=record.primary_cwe,
                gold_cwes=record.cwe_ids,
            )
        )
    return examples


def build_version_examples(
    records: Iterable[CVERecord],
    *,
    require_version: bool = True,
) -> list[BenchmarkExample]:
    examples: list[BenchmarkExample] = []
    for record in records:
        if require_version and not record.version_mentions:
            continue
        examples.append(
            BenchmarkExample(
                task=TaskName.VERSION_PARSING,
                cve_id=record.cve_id,
                prompt=build_version_prompt(record),
                gold_versions=record.version_mentions,
            )
        )
    return examples


def build_all_examples(
    records: Iterable[CVERecord],
    *,
    require_cwe: bool = True,
    require_version: bool = True,
) -> list[BenchmarkExample]:
    records = list(records)
    return build_cwe_examples(records, require_cwe=require_cwe) + build_version_examples(
        records, require_version=require_version
    )
