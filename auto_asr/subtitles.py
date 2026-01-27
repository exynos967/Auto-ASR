from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SubtitleLine:
    start_s: float
    end_s: float
    text: str


def _clamp_non_negative(value: float) -> float:
    return value if value > 0 else 0.0


def _seconds_to_milliseconds(seconds: float) -> int:
    # Use rounding to avoid drifting when adding offsets chunk-by-chunk.
    return round(_clamp_non_negative(seconds) * 1000.0)


def format_srt_timestamp(seconds: float) -> str:
    total_ms = _seconds_to_milliseconds(seconds)
    ms = total_ms % 1000
    total_s = total_ms // 1000
    s = total_s % 60
    total_m = total_s // 60
    m = total_m % 60
    h = total_m // 60
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def format_vtt_timestamp(seconds: float) -> str:
    total_ms = _seconds_to_milliseconds(seconds)
    ms = total_ms % 1000
    total_s = total_ms // 1000
    s = total_s % 60
    total_m = total_s // 60
    m = total_m % 60
    h = total_m // 60
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def _normalize_text(text: str) -> str:
    # Whisper-like outputs sometimes include leading spaces; keep internal newlines.
    return "\n".join(line.strip() for line in (text or "").splitlines()).strip()


def compose_srt(lines: list[SubtitleLine]) -> str:
    out: list[str] = []
    for idx, line in enumerate(lines, start=1):
        text = _normalize_text(line.text)
        if not text:
            continue

        start_s = _clamp_non_negative(line.start_s)
        end_s = _clamp_non_negative(line.end_s)
        if end_s <= start_s:
            end_s = start_s + 0.001

        out.append(str(idx))
        out.append(f"{format_srt_timestamp(start_s)} --> {format_srt_timestamp(end_s)}")
        out.append(text)
        out.append("")
    return "\n".join(out).rstrip() + "\n"


def compose_vtt(lines: list[SubtitleLine]) -> str:
    out: list[str] = ["WEBVTT", ""]
    for line in lines:
        text = _normalize_text(line.text)
        if not text:
            continue

        start_s = _clamp_non_negative(line.start_s)
        end_s = _clamp_non_negative(line.end_s)
        if end_s <= start_s:
            end_s = start_s + 0.001

        out.append(f"{format_vtt_timestamp(start_s)} --> {format_vtt_timestamp(end_s)}")
        out.append(text)
        out.append("")
    return "\n".join(out).rstrip() + "\n"


def compose_txt(full_text: str) -> str:
    return (full_text or "").strip() + "\n"
