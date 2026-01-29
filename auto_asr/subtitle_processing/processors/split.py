from __future__ import annotations

import math

from auto_asr.subtitles import SubtitleLine


def split_text_by_delimiter(text: str, delimiter: str = "<br>") -> list[str]:
    raw = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    parts = [p.strip() for p in raw.split(delimiter)]
    return [p for p in parts if p]


def _segment_weight(text: str) -> int:
    # Use a simple character-based weight to allocate time.
    cleaned = "".join((text or "").split())
    return max(1, len(cleaned))


def split_line_to_cues(line: SubtitleLine, parts: list[str]) -> list[SubtitleLine]:
    """Split one cue into multiple cues and allocate timestamps proportionally."""
    if not parts:
        return [line]

    start_s = float(line.start_s)
    end_s = float(line.end_s)
    if end_s <= start_s:
        end_s = start_s + 0.001

    total_ms = max(1, int(round((end_s - start_s) * 1000.0)))
    weights = [_segment_weight(p) for p in parts]
    total_w = sum(weights)

    out: list[SubtitleLine] = []
    cur_ms = 0
    for i, (w, txt) in enumerate(zip(weights, parts, strict=False)):
        if i == len(parts) - 1:
            next_ms = total_ms
        else:
            next_ms = cur_ms + int(math.floor(total_ms * (w / total_w)))

        seg_start_s = start_s + (cur_ms / 1000.0)
        seg_end_s = start_s + (next_ms / 1000.0)
        if seg_end_s <= seg_start_s:
            seg_end_s = seg_start_s + 0.001

        out.append(SubtitleLine(start_s=seg_start_s, end_s=seg_end_s, text=txt))
        cur_ms = next_ms

    # Ensure last end matches original end (avoid rounding drift).
    last = out[-1]
    out[-1] = SubtitleLine(start_s=last.start_s, end_s=end_s, text=last.text)
    return out


__all__ = ["split_line_to_cues", "split_text_by_delimiter"]

