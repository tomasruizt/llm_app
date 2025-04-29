from pathlib import Path
from llmlib.gemini.gemini_code import (
    BatchEntry,
    GeminiAPI,
    GeminiModels,
    cache_content,
    chunk,
    create_client,
    get_cached_content,
)
from datetime import datetime
from google.genai.types import CachedContent
import pytest

from tests.helpers import (
    assert_model_knows_capital_of_france,
    assert_model_recognizes_afd_in_video,
    assert_model_recognizes_pyramid_in_image,
    assert_model_supports_multiturn,
    assert_model_supports_multiturn_with_6min_video,
    assert_model_supports_multiturn_with_multiple_imgs,
    file_for_test,
    is_ci,
    video_file,
)


@pytest.mark.skipif(condition=is_ci(), reason="Avoid costs")
def test_gemini_vision_using_interface():
    model = GeminiAPI(model_id=GeminiModels.gemini_20_flash_lite, max_output_tokens=50)
    assert_model_knows_capital_of_france(model)
    assert_model_recognizes_pyramid_in_image(model)
    assert_model_recognizes_afd_in_video(model)


@pytest.mark.skipif(condition=is_ci(), reason="Avoid costs")
def test_gemini_location():
    model = GeminiAPI(
        model_id=GeminiModels.gemini_25_pro,
        location="us-central1",
        max_output_tokens=100,
    )
    assert_model_knows_capital_of_france(model)


@pytest.mark.skipif(condition=is_ci(), reason="Avoid costs")
def test_multiturn_textonly_conversation():
    model = GeminiAPI(model_id=GeminiModels.gemini_20_flash_lite, max_output_tokens=50)
    assert_model_supports_multiturn(model)


@pytest.mark.skipif(condition=is_ci(), reason="Avoid costs")
@pytest.mark.parametrize("use_context_caching", [False, True])
def test_multiturn_conversation_with_6min_video_and_context_caching(
    use_context_caching: bool,
):
    """
    Context caching is supported only for Gemini 1.5 Pro and Flash
    https://cloud.google.com/vertex-ai/generative-ai/docs/context-cache/context-cache-overview#supported_models
    """
    model = GeminiAPI(
        model_id=GeminiModels.gemini_15_flash,
        max_output_tokens=50,
        use_context_caching=use_context_caching,
        delete_files_after_use=False,
    )
    assert_model_supports_multiturn_with_6min_video(model)


@pytest.mark.skipif(condition=is_ci(), reason="Avoid costs")
@pytest.mark.parametrize("use_context_caching", [False, True])
def test_gemini_multiturn_convo_with_multiple_imgs(use_context_caching: bool):
    model = GeminiAPI(
        model_id=GeminiModels.gemini_15_flash,
        max_output_tokens=100,
        use_context_caching=use_context_caching,
        delete_files_after_use=False,
    )
    assert_model_supports_multiturn_with_multiple_imgs(model)


@pytest.mark.skipif(condition=is_ci(), reason="Avoid costs")
def test_get_cached_content():
    """We can cache content and reuse the cache later"""
    path: Path = file_for_test("tasting travel - rome italy.mp4")
    client = create_client()
    model_id = GeminiModels.gemini_15_flash
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
    model = GeminiAPI(model_id=GeminiModels.gemini_15_flash, max_output_tokens=500)
    batch = [
        BatchEntry(
            prompt="What do you see in each image?",
            files=[file_for_test("pyramid.jpg"), file_for_test("mona-lisa.png")],
            row_data={"post": "123", "author": "John Doe"},
        ),
        BatchEntry(
            prompt="What do you see in the video?",
            files=[video_file()],
            row_data={"post": "567", "author": "Jane Doe"},
        ),
    ]
    tgt_dir = file_for_test(f"batch/{datetime.now().strftime('%Y%m%d_%H%M%S')}/")
    model.submit_batch_job(batch, tgt_dir=tgt_dir)
    assert Path(tgt_dir / "input.jsonl").exists()


def test_chunk():
    xs = [1, 2, 3, 4, 5]
    assert chunk(xs, 2) == [[1, 2], [3, 4], [5]]
    assert chunk(xs, 3) == [[1, 2, 3], [4, 5]]

    assert chunk([1], 2) == [[1]]
    assert chunk([], 2) == []
