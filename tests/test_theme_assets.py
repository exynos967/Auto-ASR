from __future__ import annotations

from pathlib import Path


def test_miku_theme_assets_are_vendored():
    root = Path(__file__).resolve().parents[1]
    theme_dir = root / "theme" / "miku"
    assert (theme_dir / "theme_schema@1.2.2.json").exists()

