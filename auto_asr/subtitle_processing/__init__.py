from .base import SubtitleProcessor, get_processor, list_processors, register_processor

# Import built-in processors for registration side effects (OCP-friendly plugin registry).
from . import processors as _processors  # noqa: F401

__all__ = ["SubtitleProcessor", "get_processor", "list_processors", "register_processor"]
