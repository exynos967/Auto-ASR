from __future__ import annotations


def test_app_uses_miku_theme():
    import app

    css = app.THEME._get_theme_css()  # type: ignore[attr-defined]
    assert "light-miku-faded.webp" in css
    assert "dark-miku.webp" in css

