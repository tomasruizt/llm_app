from llmlib.base_llm import LLM, Message
from PIL import Image
from llmlib.openai.openai_transcription import TranscriptionModel
from llmlib.rest_api.restapi_client import encode_as_png_in_base64
from llmlib.semantic_similarity import SemanticSimilarity
import pytest
from llmlib.openai.openai_completion import (
    OpenAIModel,
    as_dict,
    config_for_openrouter,
    extract_msgs,
    to_openai_gen_kwargs,
)
from deepdiff import DeepDiff

from .helpers import (
    TranscriptionCases,
    assert_model_can_answer_batch_of_text_prompts,
    assert_model_can_output_json_schema,
    assert_model_can_use_multiple_gen_kwargs_in_batch,
    assert_model_knows_capital_of_france,
    assert_model_recognizes_pyramid_in_image,
    assert_model_returns_passed_metadata,
    assert_string_almost_equal,
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


def test_candidate_count_is_translated_to_openai_n():
    kwargs = to_openai_gen_kwargs({"temperature": 1.0, "candidate_count": 3})
    assert kwargs == {"temperature": 1.0, "n": 3}


def test_as_dict_preserves_multiple_choices():
    completion = {
        "choices": [
            {"message": {"content": "first", "reasoning_content": "thinking 1"}},
            {"message": {"content": "second", "reasoning_content": "thinking 2"}},
        ],
        "usage": {"prompt_tokens": 11, "completion_tokens": 22},
    }

    result = as_dict(completion)

    assert result["response"] == ["first", "second"]
    assert result["reasoning"] == ["thinking 1", "thinking 2"]
    assert result["n_input_tokens"] == 11
    assert result["n_output_tokens"] == 22


@pytest.mark.skipif(condition=is_ci(), reason="Avoid costs")
def test_openai_vision():
    model: LLM = OpenAIModel()
    assert_model_knows_capital_of_france(model)
    assert_model_recognizes_pyramid_in_image(model)


@pytest.mark.skipif(condition=is_ci(), reason="Avoid costs")
def test_openrouter_knows_capital_of_france():
    model: LLM = OpenAIModel(
        model_id="Qwen/Qwen3-32B",
        **config_for_openrouter(),
    )
    assert_model_knows_capital_of_france(model, output_dict=True, check_thoughts=True)


def test_openrouter_can_use_multiple_gen_kwargs_in_batch():
    model: LLM = OpenAIModel(
        model_id="Qwen/Qwen3-32B",
        **config_for_openrouter(),
    )
    assert_model_can_use_multiple_gen_kwargs_in_batch(model)


def test_openai_can_answer_batch_of_text_prompts():
    model: LLM = OpenAIModel()
    assert_model_can_answer_batch_of_text_prompts(model)


def test_openai_returns_passed_metadata():
    model: LLM = OpenAIModel()
    assert_model_returns_passed_metadata(model)


def test_openai_can_output_json_schema():
    model: LLM = OpenAIModel()
    assert_model_can_output_json_schema(model)


def test_transcription_openai():
    model = TranscriptionModel()
    cases = [TranscriptionCases.librispeech_2, TranscriptionCases.afd_video]
    files = [case.file for case in cases]
    assert all(f.exists() for f in files)

    transcriptions = model.transcribe_batch(files)
    for actual, case in zip(transcriptions, cases):
        assert_string_almost_equal(actual, case.expected_transcription)


def test_translation(semantic_similarity: SemanticSimilarity):
    model = TranscriptionModel()
    case = TranscriptionCases.afd_video
    translation = model.transcribe_batch([case.file], translate=True)
    similarity = semantic_similarity.cos_sim(
        translation[0], case.english_translation_en
    )
    assert similarity > 0.8


@pytest.fixture(scope="session")
def semantic_similarity():
    yield SemanticSimilarity()
