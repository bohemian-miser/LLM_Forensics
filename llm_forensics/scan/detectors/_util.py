"""Helpers shared by the built-in adapters."""
from __future__ import annotations


def normalise(scores: dict[int, float], floor: float = 0.0) -> dict[int, float]:
    """Map raw per-layer anomaly scores to 0..1 by dividing by max(floor, max score)."""
    if not scores:
        return {}
    top = max(max(scores.values()), floor)
    if top <= 0:
        return {int(l): 0.0 for l in scores}
    return {int(l): max(0.0, min(1.0, float(v) / top)) for l, v in scores.items()}


def best_edit(report: dict) -> dict | None:
    eds = (report or {}).get("suspected_edits") or []
    return eds[0] if eds else None
