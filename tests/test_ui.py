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
        assert any("run:" in c.value for c in at.code)            # the YAML under "Advanced"
        assert any("Estimated cost" in m.label for m in at.metric)
    else:
        assert any("engineer-days" in m.label.lower() for m in at.metric)


def test_every_input_has_help_text():
    """The reader is a product manager: every control that takes a decision carries a help icon."""
    at = AppTest.from_file(APP, default_timeout=60)
    at.run()
    for page in ("New run", "Migration calculator"):
        at.sidebar.radio[0].set_value(page).run()
        assert not at.exception, at.exception
        widgets = [*at.selectbox, *at.multiselect, *at.number_input, *at.checkbox, *at.text_input, *at.text_area]
        widgets += [w for w in at.radio if w.label != "Page"]
        missing = [w.label for w in widgets if not getattr(w, "help", None)]
        assert missing == [], f"{page}: inputs without help: {missing}"


def test_new_run_uses_plain_language():
    at = AppTest.from_file(APP, default_timeout=60)
    at.run()
    at.sidebar.radio[0].set_value("New run").run()
    text = " ".join(h.value for h in at.subheader) + " ".join(c.label for c in at.checkbox)
    for jargon in ("Cells", "t1-g6", "Pairing", "TTFT", "TPOT", "DSPM preset"):
        assert jargon not in text, f"{jargon!r} is shown to the user without explanation"
    assert "GPU · NVIDIA" in text and "CPU · Intel Xeon with AMX" in text
    # the DSPM suite shows scenarios and the speed requirement in words (AppTest exposes the
    # formatted labels as options)
    at.radio[0].set_value("dspm").run()
    assert not at.exception, at.exception
    labels = [o for w in at.multiselect for o in w.options] + [o for w in at.radio for o in w.options]
    assert any(o.startswith("Bulk document scan: Sensitive") for o in labels)
    assert any(o.startswith("Inline, strict") for o in labels)
    assert not any(o in ("W1", "inline_strict", "t1-g6", "bf16") for o in labels)
    # the YAML still carries the technical values
    yaml_text = next(c.value for c in at.code if "run:" in c.value)
    assert "suite: dspm" in yaml_text and "- W1" in yaml_text and "sla_preset: inline_standard" in yaml_text
