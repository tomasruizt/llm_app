from llmlib.whisper import Whisper
import pytest
from tests.helpers import is_ci, test_file_path


@pytest.mark.skipif(condition=is_ci(), reason="No GPU in CI")
def test_transcription():
    file = str(test_file_path(name="some-audio.flac"))  # Librispeech sample 2
    expected_transcription = "before he had time to answer a much encumbered vera burst into the room with the question i say can i leave these here these were a small black pig and a lusty specimen of black-red game-cock"
    model = Whisper()
    actual_transcription: str = model.transcribe_file(file)
    assert actual_transcription == expected_transcription
