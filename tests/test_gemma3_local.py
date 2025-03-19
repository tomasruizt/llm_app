from llmlib.gemma3_local import Gemma3Local
import pytest
from .helpers import (
    assert_model_recognizes_pyramid_in_image,
    assert_model_supports_multiturn_with_6min_video,
    is_ci,
    assert_model_knows_capital_of_france,
    assert_model_supports_multiturn,
    assert_model_supports_multiturn_with_multiple_imgs,
)


cls = Gemma3Local


@pytest.fixture(scope="session")
def gemma3():
    # 4B model needs 12GB VRAM for 20 frames
    return cls(model_id="google/gemma-3-4b-it", max_n_frames_per_video=20)


def test_gemma3_local_warnings():
    warnings = cls.get_warnings()
    assert len(warnings) == 0


@pytest.mark.skipif(condition=is_ci(), reason="Avoid costs")
def test_gemma3_local_complete_msgs_text_only(gemma3):
    assert_model_knows_capital_of_france(gemma3)


@pytest.mark.skipif(condition=is_ci(), reason="Avoid costs")
def test_gemma3_local_complete_msgs_with_image(gemma3):
    assert_model_recognizes_pyramid_in_image(gemma3)


@pytest.mark.skipif(condition=is_ci(), reason="Avoid costs")
def test_gemma3_local_multi_turn_text_conversation(gemma3):
    assert_model_supports_multiturn(gemma3)


@pytest.mark.skipif(condition=is_ci(), reason="Avoid costs")
def test_gemma3_local_multi_turn_with_images(gemma3):
    assert_model_supports_multiturn_with_multiple_imgs(gemma3)


@pytest.mark.skipif(condition=is_ci(), reason="Avoid costs")
def test_gemma3_local_multi_turn_with_6min_video(gemma3):
    assert_model_supports_multiturn_with_6min_video(gemma3)
