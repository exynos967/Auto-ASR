from auto_asr.subtitles import (
    SubtitleLine,
    compose_srt,
    compose_vtt,
    format_srt_timestamp,
    format_vtt_timestamp,
)


def test_format_srt_timestamp():
    assert format_srt_timestamp(0.0) == "00:00:00,000"
    assert format_srt_timestamp(1.234) == "00:00:01,234"
    assert format_srt_timestamp(60.0) == "00:01:00,000"
    assert format_srt_timestamp(3661.2) == "01:01:01,200"


def test_format_vtt_timestamp():
    assert format_vtt_timestamp(0.0) == "00:00:00.000"
    assert format_vtt_timestamp(1.234) == "00:00:01.234"
    assert format_vtt_timestamp(60.0) == "00:01:00.000"
    assert format_vtt_timestamp(3661.2) == "01:01:01.200"


def test_compose_srt_basic():
    srt = compose_srt(
        [
            SubtitleLine(start_s=0.0, end_s=1.0, text="hello"),
            SubtitleLine(start_s=1.5, end_s=2.0, text="world"),
        ]
    )
    assert "1\n00:00:00,000 --> 00:00:01,000\nhello\n" in srt
    assert "2\n00:00:01,500 --> 00:00:02,000\nworld\n" in srt


def test_compose_vtt_header():
    vtt = compose_vtt([SubtitleLine(start_s=0.0, end_s=1.0, text="hi")])
    assert vtt.startswith("WEBVTT\n\n")
