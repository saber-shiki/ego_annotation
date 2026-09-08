"""CFR-only video timing contracts, using decoded presentation timestamps.

Frame counts and average FPS alone do not establish CFR. Integer PTS are checked
against the exact rational cadence. Fractional tick rounding is allowed only
where that cadence is not exactly representable in the source time base; there
is no half-frame allowance. Annotation timestamps use seconds and must match
the nominal grid within one microsecond of numerical/decimal precision.

This module is stdlib-only so launch preflight can use it before model imports.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import json
import math
from pathlib import Path
import subprocess

PACKAGE_TIME_TOLERANCE_S = 1e-6


@dataclass(frozen=True)
class CfrVideoTiming:
    frame_rate: Fraction
    time_base: Fraction
    presentation_ticks: tuple[int, ...]
    last_duration_ticks: int
    source_size_wh: tuple[int, int]
    rate_field: str
    max_grid_error_s: float

    @property
    def fps(self) -> float:
        return float(self.frame_rate)

    @property
    def presentation_times_s(self) -> tuple[float, ...]:
        return tuple(float(t * self.time_base) for t in self.presentation_ticks)

    def report(self) -> dict:
        return {
            "fps_ratio": str(self.frame_rate),
            "fps_source_field": self.rate_field,
            "time_base_s": str(self.time_base),
            "first_pts_s": self.presentation_times_s[0],
            "frame_count": len(self.presentation_ticks),
            "duration_s": float(
                (
                    self.presentation_ticks[-1]
                    - self.presentation_ticks[0]
                    + self.last_duration_ticks
                )
                * self.time_base
            ),
            "max_cfr_grid_error_s": self.max_grid_error_s,
            "validation": "decoded integer PTS against rational CFR grid; only fractional tick quantization allowed",
        }


def _integer(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise ValueError(f"missing or invalid {label}")
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"missing or invalid {label}") from exc


def _positive_ratio(value, label):
    try:
        ratio = Fraction(str(value))
    except (ValueError, ZeroDivisionError) as exc:
        raise ValueError(f"invalid {label}") from exc
    if ratio <= 0:
        raise ValueError(f"invalid {label}")
    return ratio


def _frame_pts(frame):
    # FFmpeg 4.x exposes AVFrame presentation timestamps as pkt_pts; newer
    # ffprobe uses pts. Both are explicit PTS, unlike best_effort_timestamp.
    fields = [frame[k] for k in ("pts", "pkt_pts") if frame.get(k) is not None]
    if not fields:
        raise ValueError("missing source presentation timestamp")
    values = [_integer(v, "source presentation timestamp") for v in fields]
    if len(set(values)) != 1:
        raise ValueError("conflicting source presentation timestamp aliases")
    return values[0]


def _tick_matches(ticks, expected):
    # The interval between two rounded PTS may differ by less than one clock
    # tick. At integral positions there is no quantization allowance at all.
    lower = expected.numerator // expected.denominator
    upper = -(-expected.numerator // expected.denominator)
    return lower <= ticks <= upper


def read_cfr_video_timing(path, expected_frame_count=None):
    command = [
        "ffprobe",
        "-v",
        "error",
        "-threads",
        "1",
        "-select_streams",
        "v:0",
        "-show_frames",
        "-show_streams",
        "-show_entries",
        "stream=avg_frame_rate,r_frame_rate,time_base,width,height,duration_ts:frame=pts,pkt_pts,duration,pkt_duration",
        "-of",
        "json",
        str(Path(path)),
    ]
    result = subprocess.run(
        command, check=True, capture_output=True, text=True, timeout=120
    )
    payload = json.loads(result.stdout)
    streams = payload.get("streams", [])
    frames = payload.get("frames", [])
    if len(streams) != 1 or not frames:
        raise ValueError(
            "source presentation timestamps require one nonempty video stream"
        )
    if expected_frame_count is not None and len(frames) != expected_frame_count:
        raise ValueError("video/package decoded presentation frame count mismatch")
    stream = streams[0]
    rate_field = "avg_frame_rate"
    if stream.get(rate_field) in (None, "0/0", "0/1"):
        rate_field = "r_frame_rate"
    rate = _positive_ratio(stream.get(rate_field), "source video frame rate")
    time_base = _positive_ratio(stream.get("time_base"), "source video time base")
    pts = tuple(_frame_pts(f) for f in frames)
    if any(b <= a for a, b in zip(pts, pts[1:])):
        raise ValueError("source presentation timestamps must be strictly increasing")
    step = 1 / (rate * time_base)
    errors = []
    for i, value in enumerate(pts):
        relative = value - pts[0]
        expected = i * step
        if not _tick_matches(relative, expected):
            raise ValueError(
                f"source presentation timestamps are not CFR at frame {i}; VFR is unsupported"
            )
        errors.append(float(abs(relative - expected) * time_base))
    last_duration = frames[-1].get("duration", frames[-1].get("pkt_duration"))
    if last_duration is None:
        total = _integer(stream.get("duration_ts"), "source video duration")
        last_duration = total - (pts[-1] - pts[0])
    last_duration = _integer(last_duration, "source final frame duration")
    if last_duration <= 0 or not _tick_matches(last_duration, step):
        raise ValueError("source final frame duration does not match CFR cadence")
    width = _integer(stream.get("width"), "source width")
    height = _integer(stream.get("height"), "source height")
    if width <= 0 or height <= 0:
        raise ValueError("invalid source video dimensions")
    return CfrVideoTiming(
        rate, time_base, pts, last_duration, (width, height), rate_field, max(errors)
    )


def validate_package_cfr_times(times, frame_rate):
    values = tuple(float(t) for t in times)
    if not values or not all(math.isfinite(t) for t in values):
        raise ValueError("invalid package timestamps")
    if any(b <= a for a, b in zip(values, values[1:])):
        raise ValueError("package timestamps must be strictly increasing")
    errors = [
        abs((value - values[0]) - float(Fraction(i) / frame_rate))
        for i, value in enumerate(values)
    ]
    if max(errors) > PACKAGE_TIME_TOLERANCE_S:
        raise ValueError("package constant-rate timeline mismatch; VFR is unsupported")
    return max(errors)
