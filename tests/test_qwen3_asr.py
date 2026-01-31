from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def test_resolve_qwen3_language_maps_common_codes():
    from auto_asr.qwen3_asr import resolve_qwen3_language

    assert resolve_qwen3_language(None) is None
    assert resolve_qwen3_language("auto") is None
    assert resolve_qwen3_language("zh") == "Chinese"
    assert resolve_qwen3_language("en") == "English"
    assert resolve_qwen3_language("ja") == "Japanese"


def test_transcribe_chunks_qwen3_builds_segments_from_time_stamps(monkeypatch):
    from auto_asr.qwen3_asr import Qwen3ASRConfig, transcribe_chunks_qwen3

    @dataclass
    class R:
        language: str
        text: str
        time_stamps: object | None = None

    class FakeModel:
        def transcribe(self, *, audio, language, return_time_stamps):
            # Qwen3-ASR timeline is derived from Silero VAD in our pipeline; we never
            # request forced-aligned timestamps from the model.
            assert return_time_stamps is False
            assert isinstance(audio, list) and len(audio) == 1
            assert language == ["English"]
            return [R(language="en", text="hello world")]

    def fake_make_model(_cfg):
        return FakeModel()

    monkeypatch.setattr("auto_asr.qwen3_asr._make_model", fake_make_model)

    wav = np.zeros(16000, dtype=np.float32)
    cfg = Qwen3ASRConfig(
        model="Qwen/Qwen3-ASR-1.7B",
        device="cpu",
    )
    out = transcribe_chunks_qwen3(
        chunks=[wav],
        cfg=cfg,
        language="en",
        sample_rate=16000,
    )
    assert len(out) == 1
    assert out[0].text == "hello world"
    assert out[0].segments == []


def test_preload_qwen3_model_can_disable_forced_aligner(monkeypatch):
    import auto_asr.qwen3_asr as qwen3_asr

    qwen3_asr._MODEL_CACHE.clear()

    seen = {}

    class FakeQwen3ASRModel:
        @classmethod
        def from_pretrained(cls, model, **kwargs):  # type: ignore[no-untyped-def]
            seen["model"] = model
            seen["kwargs"] = dict(kwargs)
            return object()

    monkeypatch.setattr(qwen3_asr, "_import_qwen_asr", lambda: FakeQwen3ASRModel)
    monkeypatch.setattr(qwen3_asr, "configure_model_cache_env", lambda: None)

    qwen3_asr.preload_qwen3_model(
        qwen3_asr.Qwen3ASRConfig(
            model="Qwen/Qwen3-ASR-1.7B",
            device="cpu",
            max_inference_batch_size=1,
        )
    )

    assert seen["model"] == "Qwen/Qwen3-ASR-1.7B"
    assert "forced_aligner" not in seen["kwargs"]
    assert "forced_aligner_kwargs" not in seen["kwargs"]

    qwen3_asr._MODEL_CACHE.clear()
