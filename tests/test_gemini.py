from llmlib.gemini.gemini_code import GeminiAPI, GeminiModels
import pytest

from tests.helpers import (
    assert_model_knows_capital_of_france,
    assert_model_recognizes_afd_in_video,
    assert_model_recognizes_pyramid_in_image,
    assert_model_supports_multiturn,
    assert_model_supports_multiturn_with_file,
    is_ci,
)


@pytest.mark.skipif(condition=is_ci(), reason="Avoid costs")
def test_gemini_vision_using_interface():
    model = GeminiAPI(model_id=GeminiModels.gemini_20_flash_lite, max_output_tokens=50)
    assert_model_knows_capital_of_france(model)
    assert_model_recognizes_pyramid_in_image(model)
    assert_model_recognizes_afd_in_video(model)


@pytest.mark.skipif(condition=is_ci(), reason="Avoid costs")
def test_multiturn_conversation():
    model = GeminiAPI(model_id=GeminiModels.gemini_20_flash_lite, max_output_tokens=50)
    assert_model_supports_multiturn(model)


@pytest.mark.skipif(condition=is_ci(), reason="Avoid costs")
def test_multiturn_conversation_with_file():
    model = GeminiAPI(model_id=GeminiModels.gemini_20_flash_lite, max_output_tokens=50)
    assert_model_supports_multiturn_with_file(model)
