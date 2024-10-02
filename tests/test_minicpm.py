from llmlib.minicpm import MiniCPM
import pytest
from .helpers import (
    assert_model_knows_capital_of_france,
    assert_model_recognizes_afd_in_video,
    assert_model_recognizes_pyramid_in_image,
    is_ci,
)


@pytest.mark.skipif(condition=is_ci(), reason="No GPU in CI")
def test_minicpm_vision():
    model = MiniCPM()
    assert_model_knows_capital_of_france(model)
    assert_model_recognizes_pyramid_in_image(model)
    assert_model_recognizes_afd_in_video(model)
