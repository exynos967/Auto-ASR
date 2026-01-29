from __future__ import annotations

from auto_asr.subtitles import SubtitleLine
from auto_asr.subtitle_processing.processors.split import split_line_to_cues, split_text_by_delimiter


def test_split_text_by_delimiter_basic():
    assert split_text_by_delimiter("a<br>b") == ["a", "b"]


def test_split_line_to_cues_allocates_time_proportionally():
    line = SubtitleLine(start_s=0.0, end_s=10.0, text="a<br>bb")
    cues = split_line_to_cues(line, ["a", "bb"])
    assert len(cues) == 2
    assert cues[0].start_s == 0.0
    assert cues[-1].end_s == 10.0
    # a : bb => 1/3 : 2/3
    assert abs(cues[0].end_s - 3.333) < 0.01
    assert abs(cues[1].start_s - 3.333) < 0.01

