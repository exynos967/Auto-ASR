"""
Audio loading and (optional) VAD-based splitting utilities.

This module includes code derived from Qwen3-ASR-Toolkit (MIT License).
See `THIRD_PARTY_NOTICES.md` for license details.
"""

from __future__ import annotations

import io
import os
import subprocess

import numpy as np
import soundfile as sf

try:
    from imageio_ffmpeg import get_ffmpeg_exe  # type: ignore
except Exception:  # pragma: no cover
    get_ffmpeg_exe = None  # type: ignore[assignment]

try:
    from silero_vad import get_speech_timestamps  # type: ignore
except Exception:  # pragma: no cover
    get_speech_timestamps = None  # type: ignore[assignment]


WAV_SAMPLE_RATE = 16000


def _ffmpeg_bin() -> str:
    if get_ffmpeg_exe is None:
        return "ffmpeg"
    try:
        return get_ffmpeg_exe()
    except Exception:  # pragma: no cover
        return "ffmpeg"


def load_audio(file_path: str) -> np.ndarray:
    if file_path.startswith(("http://", "https://")):
        raise ValueError("暂不支持远程 URL，请先下载到本地文件。")

    command = [
        _ffmpeg_bin(),
        "-i",
        file_path,
        "-ar",
        str(WAV_SAMPLE_RATE),
        "-ac",
        "1",
        "-c:a",
        "pcm_s16le",
        "-f",
        "wav",
        "-",
    ]

    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except FileNotFoundError as e:
        raise RuntimeError("未找到 ffmpeg，请安装 `imageio-ffmpeg` 或系统 ffmpeg。") from e

    stdout_data, stderr_data = process.communicate()
    if process.returncode != 0:
        msg = stderr_data.decode("utf-8", errors="ignore")
        raise RuntimeError(f"ffmpeg 处理音频失败：{msg}")

    with io.BytesIO(stdout_data) as data_io:
        wav_data, _sr = sf.read(data_io, dtype="float32")

    if wav_data.ndim == 2:
        wav_data = wav_data.mean(axis=1)
    return wav_data


def process_vad(
    wav: np.ndarray,
    worker_vad_model: object | None,
    segment_threshold_s: int = 120,
    max_segment_threshold_s: int = 180,
) -> list[tuple[int, int, np.ndarray]]:
    """
    Segment long audio using Silero VAD timestamps when available, otherwise fall back
    to fixed-size chunking.
    """

    try:
        if worker_vad_model is None or get_speech_timestamps is None:
            raise RuntimeError("VAD model not available.")

        vad_params = {
            "sampling_rate": WAV_SAMPLE_RATE,
            "return_seconds": False,
            "min_speech_duration_ms": 1500,
            "min_silence_duration_ms": 500,
        }

        speech_timestamps = get_speech_timestamps(wav, worker_vad_model, **vad_params)
        if not speech_timestamps:
            raise ValueError("No speech segments detected by VAD.")

        potential_split_points = {0, len(wav)}
        for ts in speech_timestamps:
            start_of_next = int(ts["start"])
            potential_split_points.add(start_of_next)
        sorted_potential_splits = sorted(potential_split_points)

        final_split_points = {0, len(wav)}
        segment_threshold_samples = int(segment_threshold_s) * WAV_SAMPLE_RATE
        target = segment_threshold_samples
        while target < len(wav):
            closest_point = min(sorted_potential_splits, key=lambda p: abs(p - target))
            final_split_points.add(int(closest_point))
            target += segment_threshold_samples
        final_ordered_splits = sorted(final_split_points)

        max_segment_threshold_samples = int(max_segment_threshold_s) * WAV_SAMPLE_RATE
        split_points: list[float] = [0.0]

        for i in range(1, len(final_ordered_splits)):
            start = final_ordered_splits[i - 1]
            end = final_ordered_splits[i]
            segment_length = end - start

            if segment_length <= max_segment_threshold_samples:
                split_points.append(float(end))
                continue

            num_subsegments = int(np.ceil(segment_length / max_segment_threshold_samples))
            subsegment_length = segment_length / num_subsegments
            for j in range(1, num_subsegments):
                split_points.append(start + j * subsegment_length)
            split_points.append(float(end))

        segmented_wavs: list[tuple[int, int, np.ndarray]] = []
        for i in range(len(split_points) - 1):
            start_sample = int(split_points[i])
            end_sample = int(split_points[i + 1])
            segmented_wavs.append((start_sample, end_sample, wav[start_sample:end_sample]))
        return segmented_wavs

    except Exception:
        segmented_wavs: list[tuple[int, int, np.ndarray]] = []
        total_samples = len(wav)
        max_chunk_size_samples = int(max_segment_threshold_s) * WAV_SAMPLE_RATE

        for start_sample in range(0, total_samples, max_chunk_size_samples):
            end_sample = min(start_sample + max_chunk_size_samples, total_samples)
            segment = wav[start_sample:end_sample]
            if len(segment) > 0:
                segmented_wavs.append((start_sample, end_sample, segment))
        return segmented_wavs


def save_audio_file(wav: np.ndarray, file_path: str) -> None:
    dir_name = os.path.dirname(file_path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
    sf.write(file_path, wav, WAV_SAMPLE_RATE)
