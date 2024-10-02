from llmlib.base_llm import LLM
import pytest

from llmlib.llama3.llama3_vision_70b_quantized import LLama3Vision70BQuantized
from llmlib.llama3.llama3_vision_8b import LLama3Vision8B

from .helpers import (
    assert_model_knows_capital_of_france,
    assert_model_recognizes_pyramid_in_image,
    is_ci,
)


@pytest.mark.skipif(condition=is_ci(), reason="No GPU in CI")
def test_llama_8b():
    model: LLM = LLama3Vision8B()
    assert_model_knows_capital_of_france(model)
    assert_model_recognizes_pyramid_in_image(model)


@pytest.mark.skipif(condition=is_ci(), reason="No GPU in CI")
def test_llama_70b_quantized():
    model: LLM = LLama3Vision70BQuantized()
    assert_model_knows_capital_of_france(model)
    # model cannot recognize mona lisa yet
    # assert_model_recognized_mona_lisa_in_image(model)
