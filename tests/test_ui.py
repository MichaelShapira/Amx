"""The UI renders its offline pages without errors (task 17)."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

APP = str(Path(__file__).resolve().parents[1] / "ui" / "app.py")


@pytest.mark.parametrize("page", ["New run", "Migration calculator"])
def test_offline_pages_render(page):
    at = AppTest.from_file(APP, default_timeout=60)
    at.run()
    at.sidebar.radio[0].set_value(page).run()
    assert not at.exception, at.exception
    if page == "New run":
        assert any("run:" in c.value for c in at.code)
    else:
        assert any("engineer-days" in m.label.lower() for m in at.metric)
