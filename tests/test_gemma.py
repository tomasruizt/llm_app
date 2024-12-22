from llmlib.gemma import PaliGemma2
import pytest
from .helpers import assert_model_recognizes_pyramid_in_image, is_ci


@pytest.mark.skipif(condition=is_ci(), reason="No GPU in CI")
def test_gemma_vision():
    model = PaliGemma2()
    assert_model_recognizes_pyramid_in_image(model)
