from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from auto_asr.audio_tools import WAV_SAMPLE_RATE, load_audio, process_vad, save_audio_file


@dataclass(frozen=True)
class AudioChunk:
    start_sample: int
    end_sample: int
    wav: np.ndarray

    @property
    def start_s(self) -> float:
        return self.start_sample / float(WAV_SAMPLE_RATE)

    @property
    def end_s(self) -> float:
        return self.end_sample / float(WAV_SAMPLE_RATE)

    @property
    def duration_s(self) -> float:
        return (self.end_sample - self.start_sample) / float(WAV_SAMPLE_RATE)


_VAD_MODEL: object | None = None


def get_vad_model() -> object | None:
    global _VAD_MODEL
    if _VAD_MODEL is not None:
        return _VAD_MODEL

    # Optional dependency: silero_vad pulls in torch (heavy).
    # If it's not installed, we silently fall back to fixed chunking.
    try:
        from silero_vad import load_silero_vad  # type: ignore

        _VAD_MODEL = load_silero_vad(onnx=True)
    except Exception:
        _VAD_MODEL = None
    return _VAD_MODEL


def load_and_split(
    *,
    file_path: str,
    enable_vad: bool,
    vad_segment_threshold_s: int,
    vad_max_segment_threshold_s: int,
    vad_min_duration_s: int = 180,
) -> tuple[list[AudioChunk], bool]:
    wav = load_audio(file_path)

    duration_s = len(wav) / float(WAV_SAMPLE_RATE)
    if not enable_vad or duration_s < vad_min_duration_s:
        return [AudioChunk(start_sample=0, end_sample=len(wav), wav=wav)], False

    vad_model = get_vad_model()
    parts = process_vad(
        wav,
        vad_model,
        segment_threshold_s=vad_segment_threshold_s,
        max_segment_threshold_s=vad_max_segment_threshold_s,
    )
    used_vad = vad_model is not None
    return [AudioChunk(start_sample=s, end_sample=e, wav=w) for (s, e, w) in parts], used_vad


__all__ = [
    "WAV_SAMPLE_RATE",
    "AudioChunk",
    "get_vad_model",
    "load_and_split",
    "save_audio_file",
]
