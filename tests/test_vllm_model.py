from llmlib.vllm_model import ModelvLLM
import pytest
from .helpers import (
    assert_model_recognizes_pyramid_in_image,
    is_ci,
    assert_model_knows_capital_of_france,
    assert_model_supports_multiple_imgs,
    assert_model_can_answer_batch_of_img_prompts,
)


cls = ModelvLLM


@pytest.fixture(scope="session")
def vllm_model():
    model = cls(
        # model_id="google/gemma-3-4b-it",
        model_id="Qwen/Qwen2.5-VL-3B-Instruct",
        max_n_frames_per_video=20,
        gpu_size="24GB",
    )
    yield model


def test_vllm_model_local_warnings():
    warnings = cls.get_warnings()
    assert len(warnings) == 0


# Gemma3 default params: https://huggingface.co/unsloth/gemma-3-27b-it-GGUF/blob/main/params
@pytest.mark.skipif(condition=is_ci(), reason="Avoid costs")
@pytest.mark.parametrize(
    "generate_kwargs",
    [
        {"temperature": 0.0},  # greedy-decoding
        {"temperature": 1.0},
    ],
)
def test_vllm_model_local_complete_msgs_text_only(vllm_model, generate_kwargs: dict):
    assert_model_knows_capital_of_france(vllm_model, **generate_kwargs)


@pytest.mark.skipif(condition=is_ci(), reason="Avoid costs")
def test_vllm_model_local_complete_msgs_with_image(vllm_model):
    assert_model_recognizes_pyramid_in_image(vllm_model)


@pytest.mark.skipif(condition=is_ci(), reason="Avoid costs")
def test_vllm_model_local_complete_msgs_with_multiple_imgs(vllm_model):
    assert_model_supports_multiple_imgs(vllm_model)


@pytest.mark.skipif(condition=is_ci(), reason="Avoid costs")
def test_vllm_model_can_answer_batch_of_img_prompts(vllm_model):
    assert_model_can_answer_batch_of_img_prompts(vllm_model)


# MULTITURN IS NOT SUPPORTED YET
# WE USE VLLM FOR SINGLE-QUESTION BATCH WORKLOADS
# def test_vllm_model_local_multi_turn_text_conversation(vllm_model):
#     assert_model_supports_multiturn(vllm_model)

# def test_vllm_model_local_multi_turn_with_images(vllm_model):
#     assert_model_supports_multiturn_with_multiple_imgs(vllm_model)

# def test_vllm_model_local_multi_turn_with_6min_video(vllm_model):
#     assert_model_supports_multiturn_with_6min_video(vllm_model)
