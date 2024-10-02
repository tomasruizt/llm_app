from llmlib.base_llm import Message
from PIL import Image

from llmlib.phi3.phi3 import GenConf, Phi3Vision, extract_imgs_and_dicts, pad_left
import pytest
import torch

from .helpers import (
    assert_model_can_answer_batch_of_img_prompts,
    assert_model_can_answer_batch_of_text_prompts,
    assert_model_knows_capital_of_france,
    assert_model_rejects_unsupported_batches,
    get_mona_lisa_completion,
    is_ci,
)


def test_extract_imgs_and_dicts():
    img1 = Image.new(mode="RGB", size=(1, 1))
    img2 = Image.new(mode="RGB", size=(1, 1))
    msgs = [
        a_msg(),
        a_msg(img=img1, img_name="img1"),
        a_msg(img=img2, img_name="img2"),
        a_msg(),
        a_msg(img=img1, img_name="img1"),
        a_msg(img=img2, img_name="img2"),
    ]
    images, messages = extract_imgs_and_dicts(msgs)
    assert len(images) == 2
    assert len(messages) == 6
    assert "<|image_1|>" in messages[1]["content"]
    assert "<|image_1|>" in messages[4]["content"]
    assert "<|image_2|>" in messages[5]["content"]
    assert "<|image_2|>" in messages[2]["content"]


def a_msg(img: Image.Image | None = None, img_name: str | None = None) -> Message:
    return Message(role="user", msg="", img=img, img_name=img_name)


@pytest.mark.skipif(condition=is_ci(), reason="No GPU in CI")
def test_phi3_vision(model: Phi3Vision):
    assert_model_knows_capital_of_france(model)
    answer: str = get_mona_lisa_completion(model)
    assert isinstance(answer, str)


@pytest.mark.skipif(condition=is_ci(), reason="No GPU in CI")
def test_phi3_batching(model: Phi3Vision):
    assert_model_can_answer_batch_of_text_prompts(model)
    assert_model_can_answer_batch_of_img_prompts(model)


@pytest.mark.skipif(condition=is_ci(), reason="No GPU in CI")
def test_phi3_invalid_input(model: Phi3Vision):
    assert_model_rejects_unsupported_batches(model)


@pytest.fixture(scope="module")
def model():
    yield Phi3Vision(GenConf(max_new_tokens=30))


def test_padleft():
    pad_token = -1
    seqs = [torch.tensor([1, 2, 3]), torch.tensor([4, 5]), torch.tensor([6])]
    expected = torch.tensor([[1, 2, 3], [pad_token, 4, 5], [pad_token, pad_token, 6]])
    actual = pad_left(seqs, pad_token)
    assert torch.equal(actual, expected)
