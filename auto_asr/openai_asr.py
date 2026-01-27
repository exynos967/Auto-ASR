from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from openai import OpenAI


@dataclass(frozen=True)
class ASRSegment:
    start_s: float
    end_s: float
    text: str


@dataclass(frozen=True)
class ASRResult:
    text: str
    segments: list[ASRSegment]

def make_openai_client(*, api_key: str, base_url: str | None = None) -> OpenAI:
    api_key = (api_key or "").strip()
    if not api_key:
        raise RuntimeError("请在 Web UI 中填写 OpenAI API Key。")

    base_url = (base_url or "").strip() or None
    if base_url:
        return OpenAI(api_key=api_key, base_url=base_url)
    return OpenAI(api_key=api_key)


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _extract_field(obj: Any, key: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def transcribe_file_verbose(
    client: OpenAI,
    *,
    file_path: str,
    model: str = "whisper-1",
    language: str | None = None,
    prompt: str | None = None,
) -> ASRResult:
    """
    Prefer verbose_json to get per-segment timestamps (better for SRT/VTT generation).
    Falls back to plain text if the SDK/endpoint does not support it.
    """
    base_params: dict[str, Any] = {"model": model}
    if language:
        base_params["language"] = language
    if prompt:
        base_params["prompt"] = prompt

    with open(file_path, "rb") as f:
        try:
            params = dict(base_params)
            params.update(
                {
                    "file": f,
                    "response_format": "verbose_json",
                    "timestamp_granularities": ["segment"],
                }
            )
            resp = client.audio.transcriptions.create(**params)
        except Exception:
            f.seek(0)
            resp = client.audio.transcriptions.create(file=f, **base_params)

    text = _extract_field(resp, "text", "") or ""

    raw_segments = _extract_field(resp, "segments", None)
    segments: list[ASRSegment] = []
    if raw_segments:
        for seg in raw_segments:
            segments.append(
                ASRSegment(
                    start_s=_as_float(_extract_field(seg, "start", 0.0)),
                    end_s=_as_float(_extract_field(seg, "end", 0.0)),
                    text=str(_extract_field(seg, "text", "") or ""),
                )
            )
    return ASRResult(text=text, segments=segments)
