from pathlib import Path
from llmlib.gemini.gemini_code import GeminiAPI, GeminiModels, Request
import pytest

from tests.helpers import (
    assert_model_knows_capital_of_france,
    assert_model_recognizes_afd_in_video,
    assert_model_recognizes_pyramid_in_image,
    file_for_test,
    is_ci,
)


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
        model_name=GeminiModels.gemini_20_flash,
        media_files=files,
        prompt="Describe this combined images/audio/text in detail.",
    )
    description: str = req.fetch_media_description().lower()
    assert "pyramid" in description
    assert "mona lisa" in description
    assert "horses are very fast" in description


@pytest.mark.skipif(condition=is_ci(), reason="Avoid costs")
def test_gemini_vision_using_interface():
    model = GeminiAPI(model_id=GeminiModels.gemini_20_flash_lite, max_output_tokens=50)
    assert_model_knows_capital_of_france(model)
    assert_model_recognizes_pyramid_in_image(model)
    assert_model_recognizes_afd_in_video(model)
