from llmlib.base_llm import Conversation
from llmlib.vllm_model import (
    ModelvLLM,
    Message,
    to_vllm_oai_format,
    dump_dataset_as_batch_request,
)
from pathlib import Path
import pytest
from .helpers import (
    assert_model_recognizes_pyramid_in_image,
    file_for_test,
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
        # model_id="HuggingFaceTB/SmolVLM-256M-Instruct",
        max_n_frames_per_video=10,
        gpu_size="24GB",
        enforce_eager=True,
    )
    model.get_llm()
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
def test_vllm_model_knows_capital_of_france(vllm_model, generate_kwargs: dict):
    assert_model_knows_capital_of_france(vllm_model, **generate_kwargs)


@pytest.mark.skipif(condition=is_ci(), reason="Avoid costs")
def test_vllm_model_recognizes_pyramid_in_image(vllm_model):
    assert_model_recognizes_pyramid_in_image(vllm_model)


@pytest.mark.skipif(condition=is_ci(), reason="Avoid costs")
def test_vllm_supports_multiple_imgs(vllm_model):
    assert_model_supports_multiple_imgs(vllm_model)


@pytest.mark.skipif(condition=is_ci(), reason="Avoid costs")
def test_vllm_model_can_answer_batch_of_img_prompts(vllm_model):
    assert_model_can_answer_batch_of_img_prompts(vllm_model)


def test_vllm_model_format_case1():
    convo, expected_oai_format = _vllm_oai_example_img()
    assert to_vllm_oai_format(convo) == expected_oai_format


def test_vllm_model_format_case2():
    convo, expected_oai_format = _vllm_oai_example_video()
    assert to_vllm_oai_format(convo) == expected_oai_format


def test_dump_convo_as_batch_request():
    convo1, _ = _vllm_oai_example_img()
    convo2, _ = _vllm_oai_example_video()
    tgt_jsonl = file_for_test("batch/batch_input.jsonl")

    generation_kwargs = {"temperature": 0.123}
    dump_dataset_as_batch_request(
        dataset=[convo1, convo2],
        model_id="Qwen/Qwen2.5-VL-3B-Instruct",
        tgt_jsonl=tgt_jsonl,
        **generation_kwargs,
    )
    assert tgt_jsonl.exists()


def _vllm_oai_example_video() -> tuple[Conversation, list[dict]]:
    video_path: Path = file_for_test("tasting travel - rome italy.mp4")
    convo = [
        Message(role="system", msg="You are a helpful assistant."),
        Message(role="user", msg="What is in the video?", video=video_path),
    ]
    expected_oai_format = [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What is in the video?"},
                {
                    "type": "video_url",
                    "video_url": {"url": f"file://{str(video_path.absolute())}"},
                },
            ],
        },
    ]

    return convo, expected_oai_format


def _vllm_oai_example_img() -> tuple[Conversation, list[dict]]:
    img_path: Path = file_for_test("pyramid.jpg")
    convo = [
        Message(role="system", msg="You are a helpful assistant."),
        Message(role="user", msg="What is in the image?", img=img_path),
    ]
    expected_oai_format = [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What is in the image?"},
                {
                    "type": "image_url",
                    "image_url": {"url": f"file://{str(img_path.absolute())}"},
                },
            ],
        },
    ]

    return convo, expected_oai_format


# MULTITURN IS NOT SUPPORTED YET
# WE USE VLLM FOR SINGLE-QUESTION BATCH WORKLOADS
# def test_vllm_model_local_multi_turn_text_conversation(vllm_model):
#     assert_model_supports_multiturn(vllm_model)

# def test_vllm_model_local_multi_turn_with_images(vllm_model):
#     assert_model_supports_multiturn_with_multiple_imgs(vllm_model)

# def test_vllm_model_local_multi_turn_with_6min_video(vllm_model):
#     assert_model_supports_multiturn_with_6min_video(vllm_model)
