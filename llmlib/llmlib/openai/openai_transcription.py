from pathlib import Path
from openai import OpenAI
from openai.types.audio import Translation
from openai.types.audio.transcription import Transcription
import os


class TranscriptionModel:
    def transcribe_batch(
        self, files: list[str | Path], translate: bool = False
    ) -> list[str | Exception]:
        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        if translate:
            fn = fetch_translation
        else:
            fn = fetch_transcription
        return [fn(client, f) for f in files]


def fetch_transcription(client: OpenAI, fpath: str | Path) -> str:
    with open(str(fpath), "rb") as f:
        t: Transcription = client.audio.transcriptions.create(
            file=f, model="gpt-4o-transcribe"
        )
    return t.text.strip()


def fetch_translation(client: OpenAI, fpath: str | Path) -> str:
    with open(str(fpath), "rb") as f:
        t: Translation = client.audio.translations.create(file=f, model="whisper-1")
    return t.text.strip()
