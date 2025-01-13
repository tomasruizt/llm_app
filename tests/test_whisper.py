from llmlib.whisper import Whisper, WhisperOutput
import pytest
from tests.helpers import is_ci, file_for_test
import json


@pytest.fixture(scope="module")
def model() -> Whisper:
    return Whisper()


@pytest.mark.skipif(condition=is_ci(), reason="No GPU in CI")
def test_transcription(model: Whisper):
    audio_file = str(file_for_test(name="some-audio.flac"))  # Librispeech sample 2
    expected_transcription = "before he had time to answer a much encumbered vera burst into the room with the question i say can i leave these here these were a small black pig and a lusty specimen of black-red game-cock"
    actual_transcription: str = model.transcribe_file(audio_file)
    assert actual_transcription == expected_transcription


@pytest.mark.skipif(condition=is_ci(), reason="No GPU in CI")
def test_video_transcription(model: Whisper):
    video_file = str(file_for_test("video.mp4"))
    expected_fragment = (
        "Die Unionsparteien oder deren Politiker sind heute wichtige Offiziere"
    )
    transcription: str = model.transcribe_file(video_file)
    assert expected_fragment in transcription


@pytest.mark.skipif(condition=is_ci(), reason="No GPU in CI")
def test_video_transcription_with_timestamps(model: Whisper, snapshot):
    video_file = str(file_for_test("video.mp4"))
    output: WhisperOutput = model.run_pipe(
        video_file, translate=False, return_timestamps=True
    )
    snapshot.assert_match(json.dumps(output, indent=2), "transcription.json")


@pytest.mark.skipif(condition=is_ci(), reason="No GPU in CI")
def test_translation(model: Whisper):
    german_video = str(file_for_test("video.mp4"))
    translation: str = model.transcribe_file(german_video, translate=True)
    assert "The parties and their politicians" in translation


@pytest.mark.skipif(condition=is_ci(), reason="No GPU in CI")
def test_long_video_transcription(model: Whisper):
    video_file = str(file_for_test("long-video.mp4"))
    transcription: str = model.transcribe_file(video_file)
    assert isinstance(transcription, str)
