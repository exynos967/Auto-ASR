from __future__ import annotations

import numpy as np


def test_transcribe_to_subtitles_qwen3_backend_uses_model_timestamps(tmp_path, monkeypatch):
    import auto_asr.pipeline as pipeline
    from auto_asr.openai_asr import ASRResult

    wav = np.zeros(16000, dtype=np.float32)

    def fake_load_audio(_path: str):
        return wav

    def fake_get_vad_model():
        return object()

    def fake_process_vad_speech(*_args, **_kwargs):
        # One speech region from 0.5s to 1.0s.
        return [(8000, 16000, wav[8000:16000])]

    def fake_transcribe_chunks_qwen3(**_kwargs):
        assert _kwargs["cfg"].max_inference_batch_size == 4
        assert _kwargs["language"] == "en"
        assert _kwargs["sample_rate"] == 16000
        assert len(_kwargs["chunks"]) == 1
        return [ASRResult(text="hello world", segments=[])]

    monkeypatch.setattr(pipeline, "load_audio", fake_load_audio)
    monkeypatch.setattr(pipeline, "get_vad_model", fake_get_vad_model)
    monkeypatch.setattr(pipeline, "process_vad_speech", fake_process_vad_speech)
    monkeypatch.setattr(pipeline, "transcribe_chunks_qwen3", fake_transcribe_chunks_qwen3)

    res = pipeline.transcribe_to_subtitles(
        input_audio_path="dummy.wav",
        asr_backend="qwen3asr",
        output_format="srt",
        model="whisper-1",
        language="en",
        prompt=None,
        enable_vad=False,
        vad_segment_threshold_s=120,
        vad_max_segment_threshold_s=180,
        vad_threshold=0.5,
        vad_min_speech_duration_ms=200,
        vad_min_silence_duration_ms=200,
        vad_speech_pad_ms=200,
        timeline_strategy="chunk",
        vad_speech_max_utterance_s=20,
        vad_speech_merge_gap_ms=300,
        upload_audio_format="wav",
        upload_mp3_bitrate_kbps=192,
        api_concurrency=1,
        outputs_dir=str(tmp_path),
        qwen3_model="Qwen/Qwen3-ASR-1.7B",
        qwen3_device="cpu",
        qwen3_max_inference_batch_size=4,
    )

    out_text = (tmp_path / res.subtitle_file_path.split("/")[-1]).read_text(encoding="utf-8")
    assert "00:00:00,500 --> 00:00:01,000" in out_text
    assert "hello world" in out_text
