from llmlib.whisper import Whisper, WhisperOutput
import pytest
from tests.helpers import (
    assert_string_almost_equal,
    is_ci,
    file_for_test,
    TranscriptionCases,
)
import json


@pytest.fixture(scope="module")
def model() -> Whisper:
    return Whisper(use_flash_attention=False)


@pytest.mark.skipif(condition=is_ci(), reason="No GPU in CI")
def test_transcription(model: Whisper):
    case = TranscriptionCases.librispeech_2
    actual_transcription: str = model.transcribe_file(case.file)
    assert_string_almost_equal(actual_transcription, case.expected_transcription)


@pytest.mark.skipif(condition=is_ci(), reason="No GPU in CI")
def test_video_transcription(model: Whisper):
    case = TranscriptionCases.afd_video
    expected_fragment = (
        "Die Unionsparteien oder deren Politiker sind heute wichtige Offiziere"
    )
    transcription: str = model.transcribe_file(case.file)
    assert expected_fragment in transcription


@pytest.mark.skipif(condition=is_ci(), reason="No GPU in CI")
def test_video_transcription_with_timestamps(model: Whisper, snapshot):
    case = TranscriptionCases.afd_video
    output: WhisperOutput = model.run_pipe(str(case.file), translate=False)
    snapshot.assert_match(json.dumps(output, indent=2), "transcription.json")


@pytest.mark.skip(reason="Translation is currently very bad")
def test_translation(model: Whisper):
    case = TranscriptionCases.afd_video
    translation: str = model.transcribe_file(case.file, translate=True)
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


# New batch processing tests
@pytest.mark.skipif(condition=is_ci(), reason="No GPU in CI")
def test_transcribe_batch_single_file(model: Whisper):
    """Test batch transcription with a single file"""
    case = TranscriptionCases.librispeech_2

    transcriptions = model.transcribe_batch([case.file])
    assert len(transcriptions) == 1
    assert_string_almost_equal(transcriptions[0], case.expected_transcription)


@pytest.mark.skipif(condition=is_ci(), reason="No GPU in CI")
def test_transcribe_batch_multiple_files(model: Whisper):
    """Test batch transcription with multiple files"""
    audio_case = TranscriptionCases.librispeech_2
    video_case = TranscriptionCases.afd_video

    transcriptions = model.transcribe_batch([audio_case.file, video_case.file])
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
    case = TranscriptionCases.afd_video

    transcriptions = model.transcribe_batch([case.file], translate=True)
    assert len(transcriptions) == 1
    assert isinstance(transcriptions[0], str)
    assert len(transcriptions[0]) > 0


@pytest.mark.skipif(condition=is_ci(), reason="No GPU in CI")
def test_batch_vs_single_consistency(model: Whisper):
    """Test that batch processing produces the same results as single file processing"""
    case = TranscriptionCases.librispeech_2

    single_result = model.transcribe_file(case.file)
    batch_results = model.transcribe_batch([case.file])

    assert len(batch_results) == 1
    assert batch_results[0] == single_result


@pytest.mark.skipif(condition=is_ci(), reason="No GPU in CI")
def test_batch_with_invalid_file_handling(model: Whisper):
    """Test that batch processing handles invalid files gracefully"""
    case = TranscriptionCases.librispeech_2
    invalid_file = "/path/to/nonexistent/file.mp3"

    t1, t2, t3 = model.transcribe_batch([case.file, invalid_file, case.file])
    assert isinstance(t1, str)
    assert isinstance(t2, Exception)
    assert isinstance(t3, str)
