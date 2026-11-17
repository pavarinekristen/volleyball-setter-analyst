"""The session-level question this whole adaptation is for: not "did it go in", but
"how steady is it from one set to the next". Nothing here has a basketball
equivalent — field-goal percentage is one number; this is dispersion.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SessionConsistency:
    n_sets: int
    n_scored: int                       # sets with enough data to score (excludes low-confidence ones)
    contact_height_mean_m: float | None
    contact_height_std_m: float | None
    height_deficit_mean_m: float | None
    height_deficit_std_m: float | None
    jump_rate: float | None
    lateral_mean_m: float | None
    lateral_std_m: float | None         # dispersion of the contact point, sideways from the head
    vertical_mean_m: float | None
    vertical_std_m: float | None        # dispersion of the contact point, above the head
    peak_height_mean_m: float | None
    peak_height_std_m: float | None
    exit_direction_std_deg: float | None


def _stats(values: list[float]) -> tuple[float | None, float | None]:
    arr = np.array([v for v in values if v is not None and np.isfinite(v)], float)
    if not len(arr):
        return None, None
    return round(float(arr.mean()), 3), round(float(arr.std()), 3)


def summarize(sets, *, min_confidence: float = 0.0) -> SessionConsistency:
    scored = [s for s in sets if s.confidence >= min_confidence]
    h_mean, h_std = _stats([s.contact_height_m for s in scored])
    def_mean, def_std = _stats([s.height_deficit_m for s in scored])
    lat_mean, lat_std = _stats([s.lateral_m for s in scored])
    vert_mean, vert_std = _stats([s.height_above_head_m for s in scored])
    peak_mean, peak_std = _stats([s.peak_height_post_m for s in scored])
    jumps = [s.jumped for s in scored if s.jumped is not None]
    angles = [np.degrees(np.arctan2(s.exit_direction or 0, 1)) for s in scored if s.exit_direction is not None]
    return SessionConsistency(
        n_sets=len(sets), n_scored=len(scored),
        contact_height_mean_m=h_mean, contact_height_std_m=h_std,
        height_deficit_mean_m=def_mean, height_deficit_std_m=def_std,
        jump_rate=round(sum(jumps) / len(jumps), 2) if jumps else None,
        lateral_mean_m=lat_mean, lateral_std_m=lat_std,
        vertical_mean_m=vert_mean, vertical_std_m=vert_std,
        peak_height_mean_m=peak_mean, peak_height_std_m=peak_std,
        exit_direction_std_deg=round(float(np.std(angles)), 1) if len(angles) > 1 else None,
    )
