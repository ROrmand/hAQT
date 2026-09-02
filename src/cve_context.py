"""Build CVE context for stratified analysis."""

from __future__ import annotations

from src.data_loader import CVERecord


def published_year(record: CVERecord) -> int | None:
    if not record.published or len(record.published) < 4:
        return None
    try:
        return int(record.published[:4])
    except ValueError:
        return None


def build_cve_context(records: list[CVERecord]) -> dict[str, dict]:
    """Map ``cve_id`` → severity, year, primary CWE for breakdowns."""
    out: dict[str, dict] = {}
    for record in records:
        out[record.cve_id] = {
            "severity": record.severity,
            "published_year": published_year(record),
            "primary_cwe": record.primary_cwe,
        }
    return out
