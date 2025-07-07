from llmlib.whisper import Whisper, WhisperOutput
import pytest
from tests.helpers import is_ci, file_for_test
import json


@pytest.fixture(scope="module")
def model() -> Whisper:
    return Whisper(use_flash_attention=False)


@pytest.mark.skipif(condition=is_ci(), reason="No GPU in CI")
def test_transcription(model: Whisper):
    audio_file = file_for_test(name="some-audio.flac")  # Librispeech sample 2
    expected_transcription = "before he had time to answer a much encumbered vera burst into the room with the question i say can i leave these here these were a small black pig and a lusty specimen of black-red game-cock"
    actual_transcription: str = model.transcribe_file(audio_file)
    assert actual_transcription == expected_transcription


@pytest.mark.skipif(condition=is_ci(), reason="No GPU in CI")
def test_video_transcription(model: Whisper):
    video_file = file_for_test("video.mp4")
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


@pytest.mark.skip(reason="Translation is currently very bad")
def test_translation(model: Whisper):
    german_video = file_for_test("video.mp4")
    translation: str = model.transcribe_file(german_video, translate=True)
    assert "it is necessary to be a corrective in this country" in translation


@pytest.mark.skipif(condition=is_ci(), reason="No GPU in CI")
def test_long_video_transcription(model: Whisper):
    video_file = file_for_test("long-video.mp4")
    transcription: str = model.transcribe_file(video_file)
    assert isinstance(transcription, str)


@pytest.mark.skip(reason="The target behavior is not yet fully defined")
def test_multilingual_transcription(model: Whisper):
    file = file_for_test("multilingual-snippets.mp4")
    transcription: str = model.transcribe_file(file)
    assert isinstance(transcription, str)


@pytest.mark.skip(reason="Run vLLM server")
def test_whisper_vllm():
    from openai import OpenAI

    client = OpenAI(
        api_key="EMPTY",
        base_url="http://localhost:9000/v1",
    )
    fpath = file_for_test("some-audio.flac")
    assert fpath.exists()

    expected_transcription = "before he had time to answer a much encumbered vera burst into the room with the question i say can i leave these here these were a small black pig and a lusty specimen of black-red game-cock"

    with open(str(fpath), "rb") as f:
        transcription = client.audio.transcriptions.create(
            file=f,
            model="openai/whisper-large-v3-turbo",
            language="en",
            response_format="text",
            temperature=0.0,
        )
    assert expected_transcription in transcription


# New batch processing tests
@pytest.mark.skipif(condition=is_ci(), reason="No GPU in CI")
def test_transcribe_batch_single_file(model: Whisper):
    """Test batch transcription with a single file"""
    audio_file = file_for_test("some-audio.flac")
    expected_transcription = "before he had time to answer a much encumbered vera burst into the room with the question i say can i leave these here these were a small black pig and a lusty specimen of black-red game-cock"

    transcriptions = model.transcribe_batch([audio_file])
    assert len(transcriptions) == 1
    assert transcriptions[0] == expected_transcription


@pytest.mark.skipif(condition=is_ci(), reason="No GPU in CI")
def test_transcribe_batch_multiple_files(model: Whisper):
    """Test batch transcription with multiple files"""
    audio_file = file_for_test("some-audio.flac")
    video_file = file_for_test("video.mp4")

    transcriptions = model.transcribe_batch([audio_file, video_file])
    assert len(transcriptions) == 2
    assert isinstance(transcriptions[0], str)
    assert isinstance(transcriptions[1], str)
    assert len(transcriptions[0]) > 0
    assert len(transcriptions[1]) > 0


@pytest.mark.skipif(condition=is_ci(), reason="No GPU in CI")
def test_transcribe_batch_empty_list(model: Whisper):
    """Test batch transcription with empty list"""
    transcriptions = model.transcribe_batch([])
    assert transcriptions == []


@pytest.mark.skipif(condition=is_ci(), reason="No GPU in CI")
def test_transcribe_batch_with_translation(model: Whisper):
    """Test batch transcription with translation enabled"""
    german_video = file_for_test("video.mp4")

    transcriptions = model.transcribe_batch([german_video], translate=True)
    assert len(transcriptions) == 1
    assert isinstance(transcriptions[0], str)
    assert len(transcriptions[0]) > 0


@pytest.mark.skipif(condition=is_ci(), reason="No GPU in CI")
def test_batch_vs_single_consistency(model: Whisper):
    """Test that batch processing produces the same results as single file processing"""
    audio_file = file_for_test("some-audio.flac")

    single_result = model.transcribe_file(audio_file)
    batch_results = model.transcribe_batch([audio_file])

    assert len(batch_results) == 1
    assert batch_results[0] == single_result


@pytest.mark.skipif(condition=is_ci(), reason="No GPU in CI")
def test_batch_with_invalid_file_handling(model: Whisper):
    """Test that batch processing handles invalid files gracefully"""
    audio_file = file_for_test("some-audio.flac")
    invalid_file = "/path/to/nonexistent/file.mp3"

    t1, t2, t3 = model.transcribe_batch([audio_file, invalid_file, audio_file])
    assert isinstance(t1, str)
    assert isinstance(t2, Exception)
    assert isinstance(t3, str)
