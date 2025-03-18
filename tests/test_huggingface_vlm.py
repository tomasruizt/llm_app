import pytest
from llmlib.huggingface_inference import HuggingFaceVLM, HuggingFaceVLMs
from .helpers import (
    assert_model_recognizes_pyramid_in_image,
    is_ci,
    assert_model_knows_capital_of_france,
    assert_model_supports_multiturn,
    assert_model_supports_multiturn_with_multiple_imgs,
)


@pytest.fixture
def gemma3():
    return HuggingFaceVLM(model_id=HuggingFaceVLMs.gemma_3_27b_it)


def test_huggingface_vlm_warnings():
    warnings = HuggingFaceVLM.get_warnings()
    assert len(warnings) == 0


def test_huggingface_vlm_info():
    info = HuggingFaceVLM.get_info()
    assert len(info) == 3
    assert "huggingface.co" in info[0]
    assert "text-only and image+text queries" in info[1]
    assert "multi-turn conversations" in info[2]


@pytest.mark.skipif(condition=is_ci(), reason="Avoid costs")
def test_huggingface_vlm_complete_msgs_text_only(gemma3):
    assert_model_knows_capital_of_france(gemma3)


@pytest.mark.skipif(condition=is_ci(), reason="Avoid costs")
def test_huggingface_vlm_complete_msgs_with_image(gemma3):
    assert_model_recognizes_pyramid_in_image(gemma3)


@pytest.mark.skipif(condition=is_ci(), reason="Avoid costs")
def test_huggingface_vlm_multi_turn_text_conversation(gemma3):
    assert_model_supports_multiturn(gemma3)


@pytest.mark.skipif(condition=is_ci(), reason="Avoid costs")
def test_huggingface_vlm_multi_turn_with_images(gemma3):
    assert_model_supports_multiturn_with_multiple_imgs(gemma3)
