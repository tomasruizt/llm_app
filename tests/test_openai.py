from llmlib.base_llm import LLM, Message
from PIL import Image
from llmlib.rest_api.restapi_client import encode_as_png_in_base64
import pytest
from llmlib.openai.openai_completion import (
    OpenAIModel,
    extract_msgs,
)
from deepdiff import DeepDiff

from .helpers import (
    assert_model_knows_capital_of_france,
    assert_model_recognizes_pyramid_in_image,
    is_ci,
)


def test_extract_msgs():
    img = Image.new(mode="RGB", size=(1, 1))
    msgs = [
        Message(role="user", msg="Hi"),
        Message(role="assistant", msg="Hi!"),
        Message(role="user", msg="Describe:", img=img, img_name="img1"),
    ]
    messages = extract_msgs(msgs)
    expected_msgs = [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hi!"},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe:"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{encode_as_png_in_base64(img)}",
                    },
                },
            ],
        },
    ]
    assert DeepDiff(messages, expected_msgs) == {}


@pytest.mark.skipif(condition=is_ci(), reason="Avoid costs")
def test_openai_vision():
    model: LLM = OpenAIModel()
    assert_model_knows_capital_of_france(model)
    assert_model_recognizes_pyramid_in_image(model)
