from pathlib import Path
from llmlib.gemini.media_description import Request
import pytest

from tests.helpers import file_for_test, is_ci


@pytest.mark.skipif(condition=is_ci(), reason="Avoid costs")
def test_gemini_vision():
    files: list[Path] = [
        file_for_test("pyramid.jpg"),
        file_for_test("mona-lisa.png"),
        file_for_test("some-audio.mp3"),
    ]

    for path in files:
        assert path.exists()

    req = Request(
        media_files=files, prompt="Describe this combined images/audio/text in detail."
    )
    description: str = req.fetch_media_description().lower()
    assert "pyramid" in description
    assert "mona lisa" in description
    assert "horses are very fast" in description
