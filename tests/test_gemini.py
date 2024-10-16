from pathlib import Path
from llmlib.gemini.media_description import Request
import pytest

from tests.helpers import file_for_test, is_ci


@pytest.mark.skipif(condition=is_ci(), reason="Avoid costs")
def test_gemini_vision():
    img: Path = file_for_test("pyramid.jpg")
    assert img.exists()
    req = Request(media_file=img, prompt="Describe this picture shortly.")
    description: str = req.fetch_media_description()
    assert "pyramid" in description.lower()
