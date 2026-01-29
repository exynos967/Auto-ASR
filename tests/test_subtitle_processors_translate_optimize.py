from __future__ import annotations

from auto_asr.subtitle_processing.base import ProcessorContext
from auto_asr.subtitle_processing.processors.optimize import OptimizeProcessor
from auto_asr.subtitle_processing.processors.translate import TranslateProcessor
from auto_asr.subtitles import SubtitleLine


def test_translate_processor_batches_and_preserves_timestamps():
    calls: list[dict[str, str]] = []

    def chat_json(*, system_prompt: str, payload: dict[str, str], **_kwargs):
        calls.append(payload)
        return {k: f"{v}-T" for k, v in payload.items()}

    proc = TranslateProcessor()
    ctx = ProcessorContext(chat_json=chat_json)
    lines = [
        SubtitleLine(start_s=0.0, end_s=1.0, text="a"),
        SubtitleLine(start_s=1.0, end_s=2.0, text="b"),
        SubtitleLine(start_s=2.0, end_s=3.0, text="c"),
    ]
    out = proc.process(
        lines,
        ctx=ctx,
        options={"target_language": "en", "batch_size": 2, "concurrency": 1},
    )
    assert [x.text for x in out] == ["a-T", "b-T", "c-T"]
    assert out[0].start_s == 0.0 and out[0].end_s == 1.0
    assert len(calls) == 2
    assert set(calls[0].keys()) == {"1", "2"}
    assert set(calls[1].keys()) == {"3"}


def test_optimize_processor_keeps_original_when_change_too_large():
    def chat_json(*, system_prompt: str, payload: dict[str, str], **_kwargs):
        # 1: too different -> should fallback
        # 2: small typo fix -> keep
        return {
            "1": "COMPLETELY DIFFERENT",
            "2": "hello wor1d",
        }

    proc = OptimizeProcessor()
    ctx = ProcessorContext(chat_json=chat_json)
    lines = [
        SubtitleLine(start_s=0.0, end_s=1.0, text="hello world"),
        SubtitleLine(start_s=1.0, end_s=2.0, text="hello world"),
    ]
    out = proc.process(lines, ctx=ctx, options={"batch_size": 10, "concurrency": 1})
    assert [x.text for x in out] == ["hello world", "hello wor1d"]

