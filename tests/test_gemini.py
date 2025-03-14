from llmlib.gemini.gemini_code import GeminiAPI, GeminiModels
import pytest

from tests.helpers import (
    assert_model_knows_capital_of_france,
    assert_model_recognizes_afd_in_video,
    assert_model_recognizes_pyramid_in_image,
    assert_model_supports_multiturn,
    assert_model_supports_multiturn_with_6min_video,
    assert_model_supports_multiturn_with_picture,
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
@pytest.mark.parametrize("use_context_caching", [False, True])
def test_multiturn_conversation_with_file(use_context_caching: bool):
    model = GeminiAPI(
        model_id=GeminiModels.gemini_20_flash_lite,
        max_output_tokens=50,
        use_context_caching=use_context_caching,
    )
    assert_model_supports_multiturn_with_picture(model)


@pytest.mark.skipif(condition=is_ci(), reason="Avoid costs")
def test_multiturn_conversation_with_file_and_context_caching():
    """
    Context caching is supported only for Gemini 1.5 Pro and Flash
    https://cloud.google.com/vertex-ai/generative-ai/docs/context-cache/context-cache-overview#supported_models
    """
    model = GeminiAPI(
        model_id=GeminiModels.gemini_15_flash,
        max_output_tokens=50,
        use_context_caching=True,
    )
    assert_model_supports_multiturn_with_6min_video(model)
