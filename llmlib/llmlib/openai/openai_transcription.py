from pathlib import Path
from openai import OpenAI
from openai.types.audio.transcription import Transcription
import os


class TranscriptionModel:
    def transcribe_batch_vllm(self, files: list[str | Path]) -> list[str | Exception]:
        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        return [fetch_transcriptions(client, f) for f in files]


def fetch_transcriptions(client: OpenAI, fpath: str | Path) -> str:
    with open(str(fpath), "rb") as f:
        t: Transcription = client.audio.transcriptions.create(
            file=f, model="gpt-4o-transcribe"
        )
    return t.text.strip()
