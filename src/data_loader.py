"""Parse and normalize NVD/CVE JSON records into benchmark workloads."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

# Matches common version tokens in CVE descriptions / CPE URIs.
_VERSION_RE = re.compile(
    r"(?:"
    r"version(?:s)?\s+"
    r"(?:before|prior to|through|up to|<=|<|>=|>|=)?\s*"
    r"|"
    r"(?:before|prior to|through|up to)\s+"
    r")"
    r"(v?\d+(?:\.\d+){1,4}(?:[-_][A-Za-z0-9]+)?)",
    re.IGNORECASE,
)
# CPE 2.3: cpe:2.3:part:vendor:product:version:...
_CPE_23_VERSION_RE = re.compile(
    r"cpe:2\.3:[aho]:[^:]+:[^:]+:([^:]+)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class CVERecord:
    """Normalized CVE workload item with derived gold labels."""

    cve_id: str
    description: str
    cwe_ids: tuple[str, ...]
    primary_cwe: str | None
    version_mentions: tuple[str, ...]
    severity: str | None
    published: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_nvd_json(path: str | Path) -> list[dict[str, Any]]:
    """Load raw NVD JSON 2.0 feed or a list/dict sample file."""
    path = Path(path)
    with path.open(encoding="utf-8") as f:
        payload = json.load(f)

    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        if "vulnerabilities" in payload:
            return payload["vulnerabilities"]
        if "cve" in payload or "id" in payload:
            return [payload]
    raise ValueError(f"Unrecognized NVD/CVE JSON shape in {path}")


def _english_description(cve: dict[str, Any]) -> str:
    descriptions = cve.get("descriptions") or []
    for item in descriptions:
        if item.get("lang") == "en" and item.get("value"):
            return str(item["value"]).strip()
    if descriptions and descriptions[0].get("value"):
        return str(descriptions[0]["value"]).strip()
    return ""


def _extract_cwes(cve: dict[str, Any]) -> tuple[str, ...]:
    found: list[str] = []
    for weakness in cve.get("weaknesses") or []:
        for desc in weakness.get("description") or []:
            value = (desc.get("value") or "").strip().upper()
            if value.startswith("CWE-"):
                found.append(value)
    # Preserve order, drop duplicates.
    return tuple(dict.fromkeys(found))


def _extract_versions(description: str, cve: dict[str, Any]) -> tuple[str, ...]:
    mentions: list[str] = []
    for match in _VERSION_RE.finditer(description):
        mentions.append(match.group(1).lstrip("vV"))

    configs = cve.get("configurations") or []
    blob = json.dumps(configs)
    for match in _CPE_23_VERSION_RE.finditer(blob):
        ver = match.group(1)
        if ver in {"*", "-", ""}:
            continue
        if any(ch.isdigit() for ch in ver):
            mentions.append(ver.lstrip("vV"))

    return tuple(dict.fromkeys(mentions))


def _severity(cve: dict[str, Any]) -> str | None:
    metrics = cve.get("metrics") or {}
    for key in ("cvssMetricV31", "cvssMetricV30", "cvssMetricV2"):
        entries = metrics.get(key) or []
        if not entries:
            continue
        cvss = entries[0].get("cvssData") or {}
        severity = cvss.get("baseSeverity") or entries[0].get("baseSeverity")
        if severity:
            return str(severity)
    return None


def normalize_record(raw: dict[str, Any]) -> CVERecord | None:
    """Normalize one NVD vulnerability wrapper or bare CVE object."""
    cve = raw.get("cve", raw)
    cve_id = cve.get("id")
    if not cve_id:
        return None

    description = _english_description(cve)
    cwes = _extract_cwes(cve)
    versions = _extract_versions(description, cve)

    return CVERecord(
        cve_id=str(cve_id),
        description=description,
        cwe_ids=cwes,
        primary_cwe=cwes[0] if cwes else None,
        version_mentions=versions,
        severity=_severity(cve),
        published=(cve.get("published") or None),
    )


def iter_normalized(records: Iterable[dict[str, Any]]) -> Iterator[CVERecord]:
    for raw in records:
        item = normalize_record(raw)
        if item is not None and item.description:
            yield item


def load_benchmark_dataset(
    path: str | Path,
    *,
    require_cwe: bool = False,
    limit: int | None = None,
) -> list[CVERecord]:
    """Load and normalize records for benchmarking."""
    raw_records = load_nvd_json(path)
    out: list[CVERecord] = []
    for item in iter_normalized(raw_records):
        if require_cwe and not item.primary_cwe:
            continue
        out.append(item)
        if limit is not None and len(out) >= limit:
            break
    return out


def save_normalized_jsonl(records: Iterable[CVERecord], path: str | Path) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
            count += 1
    return count
