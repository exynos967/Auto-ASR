import pytest

from auto_asr.subtitle_processing.base import get_processor


def test_unknown_processor_raises():
    with pytest.raises(KeyError):
        get_processor("nope")
