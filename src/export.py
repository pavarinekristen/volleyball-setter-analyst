"""CSV and JSON exports: one row per set, one row per frame, and the session's
consistency summary — the last one has no basketball equivalent."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np


def _num(value, digits: int = 3):
    return "" if value is None or not np.isfinite(value) else round(float(value), digits)


def write(vsets, track, info, run_dir: Path, stem: str, consistency) -> list[Path]:
    sets_csv = run_dir / f"{stem}_sets.csv"
    with sets_csv.open("w", newline="") as fh:
        out = csv.writer(fh)
        out.writerow(["set", "confidence", "contact_s", "side_on", "arm_extended_deg", "jumped",
                      "jump_height_m", "contact_height_m", "ideal_height_m", "height_deficit_m",
                      "lateral_m", "height_above_head_m", "peak_height_post_m", "exit_angle_deg",
                      "exit_direction", "calibration_source", "calibration_confidence", "reason"])
        for s in vsets:
            out.writerow([s.number, s.confidence, _num(s.contact_frame / info.fps), s.side_on,
                          _num(s.arm_extended_deg, 1), s.jumped, _num(s.jump_height_m),
                          _num(s.contact_height_m), _num(s.ideal_height_m), _num(s.height_deficit_m),
                          _num(s.lateral_m), _num(s.height_above_head_m), _num(s.peak_height_post_m),
                          _num(s.exit_angle_deg, 1), _num(s.exit_direction, 0), s.calibration_source,
                          s.calibration_confidence, s.reason])

    frames_csv = run_dir / f"{stem}_frames.csv"
    set_of = np.zeros(info.n_frames, int)
    for s in vsets:
        a, b = s.window
        set_of[a:min(b + 1, info.n_frames)] = s.number
    with frames_csv.open("w", newline="") as fh:
        out = csv.writer(fh)
        out.writerow(["frame", "t_s", "ball_x_px", "ball_y_px", "set"])
        for k in range(info.n_frames):
            out.writerow([k, _num(k / info.fps), _num(track.x[k], 1), _num(track.y[k], 1), set_of[k] or ""])

    sets_json = run_dir / "sets.json"
    sets_json.write_text(json.dumps({
        "fps": info.fps, "n_sets": len(vsets),
        "sets": [{**{k: v for k, v in asdict(s).items() if k != "path"},
                  "ball_path": [[f, round(x, 1), round(y, 1)] for f, x, y in s.path]}
                 for s in vsets],
    }, indent=2, default=_json_default))

    consistency_json = run_dir / "session_consistency.json"
    consistency_json.write_text(json.dumps(asdict(consistency), indent=2, default=_json_default))
    return [sets_csv, frames_csv, sets_json, consistency_json]


def _json_default(value):
    if isinstance(value, (np.floating, float)) and not np.isfinite(value):
        return None
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    raise TypeError(f"not JSON serializable: {value!r}")
