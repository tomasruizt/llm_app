import base64
import json
import os
from pathlib import Path
import PIL
import cv2
from llmlib.base_llm import LLM, Message, LlmReq
import numpy as np
from pydantic import BaseModel
import pytest


def assert_model_knows_capital_of_france(
    model: LLM,
    output_dict: bool = False,
    check_thoughts: bool = False,
    **generate_kwargs,
) -> None:
    response: dict | str = model.complete_msgs(
        msgs=[Message(role="user", msg="What is the capital of France?")],
        output_dict=output_dict,
        **generate_kwargs,
    )
    if output_dict:
        assert isinstance(response, dict), type(response)
        rdict: dict = response
        response: str = response["response"]
    else:
        assert isinstance(response, str), type(response)

    assert "paris" in response.lower(), response

    if check_thoughts:
        assert output_dict, "check_thoughts requires output_dict=True"
        assert "reasoning" in rdict, rdict
        assert len(rdict["reasoning"]) > 0, rdict["reasoning"]


def assert_model_returns_passed_metadata(model: LLM):
    convo1 = [Message(role="user", msg="What is the capital of France?")]
    convo2 = [Message.from_prompt("What is the capital of Germany?")]
    metadata1 = {"country": "France"}
    metadata2 = {"country": "Germany"}

    responses = model.complete_batch([convo1, convo2], metadatas=[metadata1, metadata2])
    r1, r2 = list(sorted(responses, key=lambda r: r["request_idx"]))
    assert "country" in r1, r1
    assert r1["country"] == "France", r1["country"]
    assert "country" in r2, r2
    assert r2["country"] == "Germany", r2["country"]


def assert_model_can_answer_batch_of_text_prompts(model: LLM) -> None:
    prompts = [
        "What is the capital of France?",
        "What continent is south of Europe?",
        "What are the two tallest mountains in the world?",
    ]
    batch = [[Message.from_prompt(prompt)] for prompt in prompts]
    responses = list(model.complete_batch(batch=batch))
    if isinstance(responses[0], dict):
        responses = sorted(responses, key=lambda r: r["request_idx"])
        responses = [r["response"] for r in responses]
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
    if isinstance(responses[0], dict):
        responses = sorted(responses, key=lambda r: r["request_idx"])
        responses = [r["response"] for r in responses]
    assert len(responses) == 3
    assert "pyramid" in responses[0].lower(), responses[0]
    assert "forest" in responses[1].lower(), responses[1]
    assert "fish" in responses[2].lower(), responses[2]


def assert_model_deals_graciously_with_individual_failures(model: LLM) -> None:
    batch = [
        [pyramid_message()],
        [non_existing_file_message()],
        [forest_message()],
    ]
    metadatas = [{"key": 2}, {"key": 1}, {"key": 3}]
    responses = model.complete_batch(batch=batch, metadatas=metadatas)
    responses = sorted(responses, key=lambda r: r["request_idx"])
    ok_response, fail_response, ok_response2 = responses
    assert ok_response["success"]
    assert ok_response2["success"]

    assert not fail_response["success"]
    assert fail_response["error"] is not None

    expected_metadata_vals = [m["key"] for m in metadatas]
    for r, expected_val in zip(responses, expected_metadata_vals):
        assert "key" in r, r
        assert r["key"] == expected_val, r["key"]


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


def assert_model_recognizes_afd_in_video(model: LLM, **kwargs):
    video_path = file_for_test("video.mp4")
    question = "Describe the video in english"
    convo = [Message(role="user", msg=question, video=video_path)]
    answer: str = model.complete_msgs(msgs=convo, **kwargs).lower()
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


def forest_message(load_img: bool = False) -> Message:
    img: Path = file_for_test("forest.jpg")
    if load_img:
        img = PIL.Image.open(img)
    msg = Message(
        role="user", msg="Describe what you see in the picture.", img=img, img_name=""
    )
    return msg


def fish_message(load_img: bool = False) -> Message:
    img: Path = file_for_test("fish.jpg")
    if load_img:
        img = PIL.Image.open(img)
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


def non_existing_file_message() -> Message:
    return Message(
        role="user",
        msg="What is in the picture?",
        img=file_for_test("non-existing.jpg"),
    )


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


def video_message2() -> Message:
    return Message(
        role="user",
        msg="What do you see in the video?",
        files=[video_file()],
    )


def two_imgs_message() -> Message:
    return Message(
        role="user",
        msg="What do you see in each image?",
        files=[file_for_test("pyramid.jpg"), file_for_test("mona-lisa.png")],
    )


def video_file() -> Path:
    return file_for_test("tasting travel - rome italy.mp4")


def assert_model_supports_multiturn_with_multiple_imgs(model: LLM):
    convo, answer1 = assert_model_supports_multiple_imgs(model)
    convo.append(Message(role="assistant", msg=answer1))
    convo.append(Message(role="user", msg="How are they related?"))
    answer2 = model.complete_msgs(convo).lower()
    possible_answers = ["biodiversity", "ecosystem", "habitat", "environment"]
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


def assert_model_can_output_json_schema(model: LLM, check_batch_mode: bool = True):
    class Person(BaseModel):
        name: str
        age: int

    class Group(BaseModel):
        people: list[Person]

    convo = [Message.from_prompt(prompt="Output a list of 3 people")]
    r1: str = model.complete_msgs(msgs=convo, json_schema=Group)
    group1 = json.loads(r1)
    assert Group.model_validate(group1)
    assert len(group1["people"]) == 3
    if not check_batch_mode:
        return

    responses = list(model.complete_batch(batch=[convo], json_schema=Group))
    r2: str = responses[0]["response"]
    group2 = json.loads(r2)
    assert Group.model_validate(group2)
    assert len(group2["people"]) == 3


def assert_model_can_use_multiple_gen_kwargs(model: LLM):
    req1 = LlmReq(
        convo=[Message.from_prompt(prompt="Whats the capital of France?")],
        gen_kwargs={"temperature": 0.5},
    )
    req2 = LlmReq(
        convo=[Message.from_prompt(prompt="Whats the capital of Germany?")],
        gen_kwargs={"temperature": 0.8},
    )
    responses = model.complete_batchof_reqs(batch=[req1, req2])
    r1, r2 = list(sorted(responses, key=lambda r: r["request_idx"]))

    assert "temperature" in r1, r1
    assert r1["temperature"] == 0.5, r1["temperature"]
    assert "temperature" in r2, r2
    assert r2["temperature"] == 0.8, r2["temperature"]
