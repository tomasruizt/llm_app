import os
from pathlib import Path
import PIL
from llmlib.base_llm import LLM, Message
import pytest


def assert_model_knows_capital_of_france(model: LLM) -> None:
    response: str = model.complete_msgs2(
        msgs=[Message(role="user", msg="What is the capital of France?")]
    )
    assert "paris" in response.lower()


def assert_model_can_answer_batch_of_text_prompts(model: LLM) -> None:
    prompts = [
        "What is the capital of France?",
        "What continent is south of Europe?",
        "What are the two tallest mountains in the world?",
    ]
    batch = [[Message.from_prompt(prompt)] for prompt in prompts]
    responses = model.complete_batch(batch=batch)
    assert len(responses) == 3
    assert "paris" in responses[0].lower()
    assert "africa" in responses[1].lower()
    assert "everest" in responses[2].lower()


def assert_model_can_answer_batch_of_img_prompts(model: LLM) -> None:
    batch = [
        [pyramid_message()],
        [forest_message()],
        [fish_message()],
    ]
    responses = model.complete_batch(batch=batch)
    assert len(responses) == 3
    assert "pyramid" in responses[0].lower()
    assert "forest" in responses[1].lower()
    assert "fish" in responses[2].lower()


def assert_model_rejects_unsupported_batches(model: LLM) -> None:
    mixed_textonly_and_img_batch = [
        [Message.from_prompt("What is the capital of France?")],
        [pyramid_message()],
    ]
    err_msg = "Batch must contain an image in every entry or none at all."
    with pytest.raises(ValueError, match=err_msg):
        model.complete_batch(mixed_textonly_and_img_batch)


def assert_model_recognizes_pyramid_in_image(model: LLM):
    msg = pyramid_message()
    answer: str = model.complete_msgs2(msgs=[msg])
    assert "pyramid" in answer.lower()


def assert_model_recognizes_afd_in_video(model: LLM):
    video_path = file_for_test("video.mp4")
    question = "Describe the video in english"
    answer: str = model.video_prompt(video_path, question).lower()
    assert "alternative für deutschland" in answer or "afd" in answer, answer


def get_mona_lisa_completion(model: LLM) -> str:
    msg: Message = mona_lisa_message()
    answer: str = model.complete_msgs2(msgs=[msg])
    return answer


def mona_lisa_message() -> Message:
    _, img = mona_lisa_filename_and_img()
    prompt = "What is in the image?"
    msg = Message(role="user", msg=prompt, img=img, img_name="")
    return msg


def pyramid_message() -> Message:
    img_name = "pyramid.jpg"
    img = get_test_img(img_name)
    msg = Message(role="user", msg="What is in the image?", img=img, img_name="")
    return msg


def forest_message() -> Message:
    img_name = "forest.jpg"
    img = get_test_img(img_name)
    msg = Message(
        role="user", msg="Describe what you see in the picture.", img=img, img_name=""
    )
    return msg


def fish_message() -> Message:
    img_name = "fish.jpg"
    img = get_test_img(img_name)
    msg = Message(
        role="user",
        msg="What animal is depicted and where does it live?",
        img=img,
        img_name="",
    )
    return msg


def mona_lisa_filename_and_img() -> tuple[str, PIL.Image.Image]:
    img_name = "mona-lisa.png"
    img = get_test_img(img_name)
    return img_name, img


def get_test_img(name: str) -> PIL.Image.Image:
    path = file_for_test(name)
    return PIL.Image.open(path)


def file_for_test(name: str) -> Path:
    return Path(__file__).parent.parent / "test-files" / name


def is_ci() -> bool:
    is_ci_str: str = os.environ.get("CI", "false").lower()
    return is_ci_str != "false"


def assert_model_supports_multiturn(model: LLM):
    msg1 = Message.from_prompt("My name is Tomas")
    msg2 = Message.from_prompt("What is my name?")
    answer = model.complete_msgs2([msg1, msg2])
    assert "Tomas" in answer
