from pathlib import Path
import shutil
from llmlib.base_llm import LlmReq
from llmlib.gemini.gemini_code import (
    GeminiAPI,
    GeminiModels,
    cache_content,
    chunk,
    create_client,
    get_cached_content,
)
from google.genai.types import CachedContent
import pytest

from tests.helpers import (
    assert_model_can_output_json_schema,
    assert_model_can_use_multiple_gen_kwargs_in_batch,
    assert_model_knows_capital_of_france,
    assert_model_recognizes_afd_in_video,
    assert_model_recognizes_pyramid_in_image,
    assert_model_supports_multiturn,
    assert_model_supports_multiturn_with_6min_video,
    assert_model_supports_multiturn_with_multiple_imgs,
    file_for_test,
    is_ci,
    two_imgs_message,
    video_message2,
)


@pytest.mark.skipif(condition=is_ci(), reason="Avoid costs")
def test_gemini_vision_using_interface():
    model = GeminiAPI(max_output_tokens=5_000)
    assert_model_knows_capital_of_france(model)
    assert_model_recognizes_pyramid_in_image(model)
    assert_model_recognizes_afd_in_video(model)


@pytest.mark.skipif(condition=is_ci(), reason="Avoid costs")
def test_gemini_knows_capital_of_france():
    model = GeminiAPI(
        model_id=GeminiModels.gemini_25_pro,
        location="global",
        include_thoughts=True,
    )
    assert_model_knows_capital_of_france(model, check_thoughts=True, output_dict=True)


@pytest.mark.skipif(condition=is_ci(), reason="Avoid costs")
def test_gemini_can_output_json_schema():
    model = GeminiAPI()
    assert_model_can_output_json_schema(model, check_batch_mode=False)


@pytest.mark.skipif(condition=is_ci(), reason="Avoid costs")
def test_multiturn_textonly_conversation():
    model = GeminiAPI()
    assert_model_supports_multiturn(model)


@pytest.mark.skipif(condition=is_ci(), reason="Avoid costs")
@pytest.mark.parametrize("use_context_caching", [False, True])
def test_multiturn_conversation_with_6min_video_and_context_caching(
    use_context_caching: bool,
):
    """
    Context caching is supported only for specific models
    https://cloud.google.com/vertex-ai/generative-ai/docs/context-cache/context-cache-overview#supported_models
    """
    model = GeminiAPI(
        use_context_caching=use_context_caching,
        delete_files_after_use=False,
    )
    assert_model_supports_multiturn_with_6min_video(model)


@pytest.mark.skipif(condition=is_ci(), reason="Avoid costs")
@pytest.mark.parametrize("use_context_caching", [False, True])
def test_gemini_multiturn_convo_with_multiple_imgs(use_context_caching: bool):
    model = GeminiAPI(
        use_context_caching=use_context_caching,
        delete_files_after_use=False,
    )
    assert_model_supports_multiturn_with_multiple_imgs(model)


@pytest.mark.skipif(condition=is_ci(), reason="Avoid costs")
def test_get_cached_content():
    """We can cache content and reuse the cache later"""
    path: Path = file_for_test("tasting travel - rome italy.mp4")
    client = create_client()
    model_id = GeminiModels.gemini_25_flash
    _, success = get_cached_content(client, model_id=model_id, paths=[path])
    assert not success

    cache_content(client, model_id=model_id, paths=[path], ttl="60s")
    cached_content, success = get_cached_content(
        client, model_id=model_id, paths=[path]
    )
    assert success
    assert isinstance(cached_content, CachedContent)


@pytest.mark.skipif(condition=is_ci(), reason="Avoid costs")
def test_batch_mode_inference():
    model = GeminiAPI(model_id=GeminiModels.gemini_25_flash)
    batch = [
        LlmReq(
            convo=[two_imgs_message()],
            metadata={"post": "123", "author": "John Doe"},
            gen_kwargs={"temperature": 0.0},
        ),
        LlmReq(
            convo=[video_message2()],
            metadata={"post": "567", "author": "Jane Doe"},
            gen_kwargs={"temperature": 0.0},
        ),
    ]
    tgt_dir = file_for_test("unittest-batch-gemini/")
    shutil.rmtree(tgt_dir, ignore_errors=True)
    model.submit_batch_job(batch, tgt_dir=tgt_dir)
    assert Path(tgt_dir / "input.jsonl").exists()
    assert Path(tgt_dir / "submit_confirmation.json").exists()


@pytest.mark.skipif(condition=is_ci(), reason="Avoid costs")
def test_gemini_can_use_multiple_gen_kwargs():
    model = GeminiAPI(model_id=GeminiModels.gemini_25_flash)
    assert_model_can_use_multiple_gen_kwargs_in_batch(model)


def test_chunk():
    xs = [1, 2, 3, 4, 5]
    assert chunk(xs, 2) == [[1, 2], [3, 4], [5]]
    assert chunk(xs, 3) == [[1, 2, 3], [4, 5]]

    assert chunk([1], 2) == [[1]]
    assert chunk([], 2) == []
