"""The net: SAM 3.1 on one frame, optional. Unlike the basketball's rim, missing it
is not fatal — it is only a cross-check on the depth read off the player's own height.

SAM segments the net (cable, tape and mesh). The top band is read off the mask's
topmost rows; the net's pixel width across the frame gives its depth, via its
known real-world span (``NET_WIDTH_M``).
"""

from __future__ import annotations

import base64
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

CONTENT_OBJECT = "img.segment.masks"
CONTRACT = "sam31-net-v1"


@dataclass(frozen=True)
class Net:
    """The net's top band in pixels of the inference frame, or None if not found."""

    top: float          # the top cable's pixel row
    left: float
    right: float
    score: float
    frame: int

    @property
    def width(self) -> float:
        return self.right - self.left


def request_segment(client, *, model: str, image_png: bytes, prompt: str) -> tuple[dict, dict | None]:
    """Same SAM 3.1 "segment" call as the hoop's, just a different prompt."""
    b64 = base64.b64encode(image_png).decode("ascii")
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
        ]}],
        response_format={"type": "json_object"},
        extra_body={"method": "segment", "method_params": {"prompt": prompt, "mask_format": "png"}},
    )
    payload = json.loads(response.choices[0].message.content)
    usage = response.usage.model_dump() if response.usage else None
    return payload, usage


def read_frame(mp4: Path, index: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(mp4))
    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"could not read frame {index} of {mp4}")
    return frame


def cache_path(video: Path, cache_dir: Path, *, model: str, prompt: str, frame: int) -> Path:
    stat = video.stat()
    key = json.dumps({"video": video.name, "size": stat.st_size, "mtime_ns": stat.st_mtime_ns,
                      "model": model, "prompt": prompt, "frame": frame, "contract": CONTRACT},
                     sort_keys=True)
    return cache_dir / f"net.{hashlib.sha256(key.encode()).hexdigest()[:12]}.json"


def _label_map(mask) -> np.ndarray | None:
    if not isinstance(mask, dict) or mask.get("format") != "png":
        return None
    raw = base64.b64decode(str(mask.get("data", "")).split(",", 1)[-1])
    labels = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if labels is None:
        return None
    return labels[:, :, 0] if labels.ndim == 3 else labels


def locate(payload: dict, *, frame: int, width: int, height: int) -> Net | None:
    """The best-scoring net, with the top band read off its mask. ``None`` if SAM found
    nothing — the caller falls back to the player's own height for calibration."""
    content = payload.get("content")
    if not isinstance(content, dict) or content.get("object") != CONTENT_OBJECT:
        return None
    items = content.get("items") or []
    if not items:
        return None
    best = max(items, key=lambda i: float(i.get("score") or 0))

    labels = _label_map(content.get("mask"))
    instance = best.get("instance_id")
    if labels is None or instance is None:
        bx, by, bw, bh = best["bbox_xywh"]
        return Net(top=by * height, left=bx * width, right=(bx + bw) * width,
                  score=float(best.get("score") or 0), frame=frame)

    blob = labels == int(instance)
    rows = np.nonzero(blob.any(axis=1))[0]
    if not rows.size:
        return None
    # The top band: the first rows with mesh, sampled away from the sag at the
    # center so the antenna/post rows (straighter) dominate the estimate.
    top_rows = rows[:max(1, int(0.05 * len(rows)))]
    cols = np.nonzero(blob[top_rows].any(axis=0))[0]
    if not cols.size:
        return None
    return Net(top=float(top_rows.mean()), left=float(cols.min()), right=float(cols.max() + 1),
              score=float(best.get("score") or 0), frame=frame)
