import base64
import os
from pathlib import Path
import PIL
import cv2
from llmlib.base_llm import LLM, Message
import numpy as np
import pytest


def assert_model_knows_capital_of_france(model: LLM, **generate_kwargs) -> None:
    response: str = model.complete_msgs(
        msgs=[Message(role="user", msg="What is the capital of France?")],
        **generate_kwargs,
    )
    assert "paris" in response.lower(), response


def assert_model_can_answer_batch_of_text_prompts(model: LLM) -> None:
    prompts = [
        "What is the capital of France?",
        "What continent is south of Europe?",
        "What are the two tallest mountains in the world?",
    ]
    batch = [[Message.from_prompt(prompt)] for prompt in prompts]
    responses = list(model.complete_batch(batch=batch))
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
    responses = list(model.complete_batch(batch=batch))
    assert len(responses) == 3
    assert "pyramid" in responses[0].lower(), responses[0]
    assert "forest" in responses[1].lower(), responses[1]
    assert "fish" in responses[2].lower(), responses[2]


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
    answer: str = model.complete_msgs(msgs=[msg])
    assert "pyramid" in answer.lower(), answer


def assert_model_recognizes_afd_in_video(model: LLM):
    video_path = file_for_test("video.mp4")
    question = "Describe the video in english"
    answer: str = model.video_prompt(video_path, question).lower()
    assert "alternative für deutschland" in answer or "afd" in answer, answer


def get_mona_lisa_completion(model: LLM) -> str:
    msg: Message = mona_lisa_message()
    answer: str = model.complete_msgs(msgs=[msg])
    return answer


def mona_lisa_message() -> Message:
    _, img = mona_lisa_filename_and_img()
    prompt = "What is in the image?"
    msg = Message(role="user", msg=prompt, img=img, img_name="")
    return msg


def pyramid_message(load_img: bool = False) -> Message:
    img = file_for_test("pyramid.jpg")
    if load_img:
        img = PIL.Image.open(img)
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
    msg2 = Message(role="assistant", msg="Nice to meet you!")
    msg3 = Message.from_prompt("What is my name?")
    answer = model.complete_msgs([msg1, msg2, msg3])
    assert "Tomas" in answer


def assert_model_supports_multiturn_with_6min_video(model: LLM):
    convo = [video_message()]
    answer1 = model.complete_msgs(convo)
    assert "italy" in answer1.lower(), answer1

    convo.append(Message(role="assistant", msg=answer1))
    convo.append(Message(role="user", msg="What food do they eat?"))
    answer2 = model.complete_msgs(convo)
    allowed = ["lasagna", "pasta", "pizza"]  # really only lasagna, but OK
    assert any(ans in answer2.lower() for ans in allowed), answer2

    convo.append(Message(role="assistant", msg=answer2))
    convo.append(
        Message(role="user", msg="What character appears in the middle of the video?")
    )
    answer3 = model.complete_msgs(convo).lower()
    allowed = ["jesus", "michelangelo"]
    assert any(ans in answer3 for ans in allowed), answer3


def video_message() -> Message:
    video = video_file()
    return Message(role="user", msg="What country are they visiting?", video=video)


def video_file() -> Path:
    return file_for_test("tasting travel - rome italy.mp4")


def assert_model_supports_multiturn_with_multiple_imgs(model: LLM):
    convo, answer1 = assert_model_supports_multiple_imgs(model)
    convo.append(Message(role="assistant", msg=answer1))
    convo.append(Message(role="user", msg="How are they related?"))
    answer2 = model.complete_msgs(convo).lower()
    possible_answers = ["biodiversity", "ecosystem", "habitat"]
    assert any(answer in answer2 for answer in possible_answers), answer2


def assert_model_supports_multiple_imgs(model: LLM):
    files = [file_for_test("forest.jpg"), file_for_test("fish.jpg")]
    msg = Message(
        role="user", msg="Describe each image in one short sentence", files=files
    )
    convo = [msg]
    answer = model.complete_msgs(convo).lower()
    assert "forest" in answer or "river" in answer, answer
    assert "fish" in answer, answer
    return convo, answer


def decode_base64_to_array(base64_str: str) -> np.ndarray:
    """Decode base64 string to OpenCV image (numpy array)"""
    # Remove data URL prefix if present
    if "base64," in base64_str:
        base64_str = base64_str.split("base64,")[1]
    image_data = base64.b64decode(base64_str)
    np_array = np.frombuffer(image_data, np.uint8)
    image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
    return image
