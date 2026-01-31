from __future__ import annotations

import inspect

from auto_asr.funasr_asr import _extract_segments_from_result, _maybe_postprocess_text


def test_extract_segments_sentence_info_ms_scaled_to_seconds():
    res = [
        {
            "text": "hello",
            "sentence_info": [
                {"start": 0, "end": 1000, "text": "hi"},
                {"start": 1000, "end": 2500, "text": "there"},
            ],
        }
    ]
    full_text, segments = _extract_segments_from_result(res, duration_s=10.0)
    assert full_text == "hello"
    assert len(segments) == 2
    assert segments[0].start_s == 0.0
    assert segments[0].end_s == 1.0
    assert segments[0].text == "hi"
    assert segments[1].start_s == 1.0
    assert segments[1].end_s == 2.5
    assert segments[1].text == "there"


def test_extract_segments_list_of_dict_seconds_kept():
    res = [
        {"start": 0.0, "end": 1.2, "text": "a"},
        {"start": 1.2, "end": 2.0, "text": "b"},
    ]
    full_text, segments = _extract_segments_from_result(res, duration_s=30.0)
    assert full_text == "a\nb"
    assert [(s.start_s, s.end_s, s.text) for s in segments] == [
        (0.0, 1.2, "a"),
        (1.2, 2.0, "b"),
    ]


def test_maybe_postprocess_text_strips_sensevoice_rich_tags_without_gt():
    raw = "< | ja |  < | EMO _ UNKNOWN |  < | S pe ech |  < | withi tn | hello"
    assert _maybe_postprocess_text(raw) == "hello"


def test_extract_segments_timestamp_tokens_sentencepiece_like():
    res = [
        {
            "text": "< | ja |  < | EMO _ UNKNOWN |  < | S pe ech |  < | withi tn | hello world",
            "timestamp": [
                ["▁hello", 0.0, 0.5],
                ["▁world", 0.6, 1.0],
            ],
        }
    ]
    full_text, segments = _extract_segments_from_result(res, duration_s=10.0)
    assert full_text == "hello world"
    assert segments
    assert segments[0].start_s == 0.0
    assert segments[-1].end_s == 1.0


def test_extract_segments_timestamp_array_without_tokens_aligned_to_text():
    res = [
        {
            "text": "你好，世界。",
            "timestamp": [
                [0, 500],
                [600, 1000],
                [1100, 1500],
                [1600, 2000],
            ],
        }
    ]
    full_text, segments = _extract_segments_from_result(res, duration_s=10.0)
    assert full_text == "你好，世界。"
    assert len(segments) == 1
    assert segments[0].start_s == 0.0
    assert segments[0].end_s == 2.0
    assert segments[0].text == "你好，世界。"


def test_make_model_does_not_request_funasr_builtin_vad_models(monkeypatch):
    import auto_asr.funasr_asr as funasr_asr

    funasr_asr._MODEL_CACHE.clear()

    seen = {}

    def fake_automodel(**kwargs):
        seen["kwargs"] = dict(kwargs)
        return object()

    monkeypatch.setattr(funasr_asr, "_import_funasr", lambda: fake_automodel)
    monkeypatch.setattr(funasr_asr, "resolve_model_dir", lambda _m: "dummy-model")
    monkeypatch.setattr(funasr_asr, "get_remote_code_candidates", lambda **_k: [])
    monkeypatch.setattr(funasr_asr, "is_funasr_nano", lambda _m: False)

    cfg_kwargs = dict(
        model="iic/SenseVoiceSmall",
        device="cpu",
        language="auto",
        use_itn=True,
        enable_punc=True,
    )
    if "enable_vad" in getattr(funasr_asr.FunASRConfig, "__dataclass_fields__", {}):
        cfg_kwargs["enable_vad"] = True

    cfg = funasr_asr.FunASRConfig(**cfg_kwargs)  # type: ignore[arg-type]
    _ = funasr_asr._make_model(cfg)

    assert "vad_model" not in seen.get("kwargs", {})
    assert "vad_kwargs" not in seen.get("kwargs", {})


def test_transcribe_file_funasr_never_uses_merge_vad_kwargs(monkeypatch):
    import auto_asr.funasr_asr as funasr_asr

    captured = {}

    class FakeModel:
        def generate(self, **kwargs):
            captured["kwargs"] = dict(kwargs)
            return []

    monkeypatch.setattr(funasr_asr, "_make_model", lambda _cfg: FakeModel())
    monkeypatch.setattr(funasr_asr, "is_funasr_nano", lambda _m: False)

    kwargs = dict(
        file_path="dummy.wav",
        model="iic/SenseVoiceSmall",
        device="cpu",
        language="auto",
        use_itn=True,
        enable_punc=True,
        duration_s=1.0,
    )
    sig = inspect.signature(funasr_asr.transcribe_file_funasr)
    if "enable_vad" in sig.parameters:
        kwargs["enable_vad"] = True

    _ = funasr_asr.transcribe_file_funasr(**kwargs)  # type: ignore[arg-type]
    assert "merge_vad" not in captured.get("kwargs", {})
    assert "merge_length_s" not in captured.get("kwargs", {})
