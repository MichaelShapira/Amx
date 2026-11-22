"""The executive summary is rendered as Markdown by the Results page; the model's answer must reach it unwrapped."""
from webui.api.exec_summary import SYSTEM, clean_markdown

BODY = "## Bottom line\n\nThe GPU wins.\n\n| Question | GPU |\n|---|---|\n| a | 1 |"


def test_whole_answer_in_a_markdown_fence_is_unwrapped():
    assert clean_markdown("```markdown\n" + BODY + "\n```") == BODY
    assert clean_markdown("```\n" + BODY + "\n```\n") == BODY
    assert clean_markdown("  ```md  \n" + BODY + "\n   ```  ") == BODY


def test_plain_answer_and_inner_fences_are_untouched():
    assert clean_markdown(BODY + "\n") == BODY
    inner = BODY + "\n\n```\ncode\n```\n\nMore text."
    assert clean_markdown(inner) == inner           # a fence that does not wrap the whole text stays
    assert clean_markdown("") == ""


def test_prompt_forbids_the_fence():
    assert "code fence" in SYSTEM and "raw Markdown" in SYSTEM
