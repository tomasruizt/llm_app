from llmlib.gemma3_vllm import Gemma3vLLM
import pytest
from .helpers import (
    assert_model_recognizes_pyramid_in_image,
    is_ci,
    assert_model_knows_capital_of_france,
    assert_model_supports_multiple_imgs,
)


cls = Gemma3vLLM


@pytest.fixture(scope="session")
def gemma3():
    return cls(
        model_id="google/gemma-3-4b-it",
        max_n_frames_per_video=20,
        gpu_size="24GB",
    )


def test_gemma3_local_warnings():
    warnings = cls.get_warnings()
    assert len(warnings) == 0


# default params: https://huggingface.co/unsloth/gemma-3-27b-it-GGUF/blob/main/params
@pytest.mark.skipif(condition=is_ci(), reason="Avoid costs")
@pytest.mark.parametrize(
    "generate_kwargs",
    [
        {"temperature": 0.0},  # greedy-decoding
        {"temperature": 1.0},
    ],
)
def test_gemma3_local_complete_msgs_text_only(gemma3, generate_kwargs: dict):
    assert_model_knows_capital_of_france(gemma3, **generate_kwargs)


@pytest.mark.skipif(condition=is_ci(), reason="Avoid costs")
def test_gemma3_local_complete_msgs_with_image(gemma3):
    assert_model_recognizes_pyramid_in_image(gemma3)


@pytest.mark.skip(reason="currently breaks the test-runner. Not sure why")
def test_gemma3_local_complete_msgs_with_multiple_imgs(gemma3):
    assert_model_supports_multiple_imgs(gemma3)


# MULTITURN IS NOT SUPPORTED YET
# WE USE VLLM FOR SINGLE-QUESTION BATCH WORKLOADS
# def test_gemma3_local_multi_turn_text_conversation(gemma3):
#     assert_model_supports_multiturn(gemma3)

# def test_gemma3_local_multi_turn_with_images(gemma3):
#     assert_model_supports_multiturn_with_multiple_imgs(gemma3)

# def test_gemma3_local_multi_turn_with_6min_video(gemma3):
#     assert_model_supports_multiturn_with_6min_video(gemma3)
