from io import BytesIO
from llmlib.internvl import InternVL
from llmlib.minicpm import MiniCPM, to_listof_imgs
import pytest
from .helpers import (
    assert_model_knows_capital_of_france,
    assert_model_recognizes_afd_in_video,
    assert_model_recognizes_pyramid_in_image,
    file_for_test,
    is_ci,
)


@pytest.mark.skipif(condition=is_ci(), reason="No GPU in CI")
def test_minicpm_vision():
    model = MiniCPM()
    assert_model_knows_capital_of_france(model)
    assert_model_recognizes_pyramid_in_image(model)
    assert_model_recognizes_afd_in_video(model)


def test_internvl():
    model = InternVL()
    assert_model_knows_capital_of_france(model)
    assert_model_recognizes_pyramid_in_image(model)


@pytest.mark.skipif(condition=is_ci(), reason="video.mp4 does not exist in CI")
def test_to_listof_imgs():
    video_path = file_for_test("video.mp4")
    imgs = to_listof_imgs(video_path)
    assert len(imgs) >= 10
    imgs2 = to_listof_imgs(BytesIO(video_path.read_bytes()))
    assert len(imgs) == len(imgs2)
    assert imgs == imgs2
