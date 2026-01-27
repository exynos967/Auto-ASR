from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

from auto_asr.openai_asr import make_openai_client, transcribe_file_verbose
from auto_asr.subtitles import SubtitleLine, compose_srt, compose_txt, compose_vtt
from auto_asr.vad_split import WAV_SAMPLE_RATE, load_and_split, save_audio_file


@dataclass(frozen=True)
class PipelineResult:
    preview_text: str
    full_text: str
    subtitle_file_path: str
    debug: str


def _safe_stem(path: str) -> str:
    stem = Path(path).stem
    # Avoid empty/odd filenames in outputs.
    return stem if stem else "audio"


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def transcribe_to_subtitles(
    *,
    input_audio_path: str,
    openai_api_key: str,
    openai_base_url: str | None = None,
    output_format: str,
    model: str = "whisper-1",
    language: str | None = None,
    prompt: str | None = None,
    enable_vad: bool = True,
    vad_segment_threshold_s: int = 120,
    vad_max_segment_threshold_s: int = 180,
    outputs_dir: str = "outputs",
) -> PipelineResult:
    if output_format not in {"srt", "vtt", "txt"}:
        raise ValueError("output_format must be one of: srt, vtt, txt")

    client = make_openai_client(api_key=openai_api_key, base_url=openai_base_url)

    chunks, used_vad = load_and_split(
        file_path=input_audio_path,
        enable_vad=enable_vad,
        vad_segment_threshold_s=vad_segment_threshold_s,
        vad_max_segment_threshold_s=vad_max_segment_threshold_s,
    )

    subtitle_lines: list[SubtitleLine] = []
    full_text_parts: list[str] = []
    total_segments = 0

    with TemporaryDirectory(prefix="auto-asr-") as tmp_dir:
        for idx, chunk in enumerate(chunks):
            chunk_path = os.path.join(tmp_dir, f"chunk_{idx:04d}.wav")
            save_audio_file(chunk.wav, chunk_path)

            asr = transcribe_file_verbose(
                client,
                file_path=chunk_path,
                model=model,
                language=language,
                prompt=prompt,
            )

            full_text_parts.append(asr.text.strip())

            offset_s = chunk.start_sample / float(WAV_SAMPLE_RATE)
            if asr.segments:
                for seg in asr.segments:
                    subtitle_lines.append(
                        SubtitleLine(
                            start_s=offset_s + seg.start_s,
                            end_s=offset_s + seg.end_s,
                            text=seg.text,
                        )
                    )
                total_segments += len(asr.segments)
            else:
                # Fallback: coarse chunk-level subtitle.
                subtitle_lines.append(
                    SubtitleLine(
                        start_s=chunk.start_s,
                        end_s=chunk.end_s,
                        text=asr.text,
                    )
                )
                total_segments += 1

    subtitle_lines.sort(key=lambda x: (x.start_s, x.end_s))

    full_text = "\n".join([t for t in full_text_parts if t]).strip()

    if output_format == "srt":
        subtitle_text = compose_srt(subtitle_lines)
        ext = "srt"
    elif output_format == "vtt":
        subtitle_text = compose_vtt(subtitle_lines)
        ext = "vtt"
    else:
        subtitle_text = compose_txt(full_text)
        ext = "txt"

    out_base = f"{_safe_stem(input_audio_path)}-{time.strftime('%Y%m%d-%H%M%S')}"
    out_path = Path(outputs_dir) / f"{out_base}.{ext}"
    _write_text(out_path, subtitle_text)

    preview = subtitle_text[:5000]
    debug = (
        f"chunks={len(chunks)}, segments={total_segments}, "
        f"vad={'on' if enable_vad else 'off'}(used={used_vad}), "
        f"vad_segment_threshold_s={vad_segment_threshold_s}, "
        f"vad_max_segment_threshold_s={vad_max_segment_threshold_s}"
    )
    return PipelineResult(
        preview_text=preview,
        full_text=full_text,
        subtitle_file_path=str(out_path),
        debug=debug,
    )
