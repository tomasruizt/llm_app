from dataclasses import dataclass, field
import json
from logging import getLogger
from pathlib import Path
from typing import Any, TypedDict
import warnings
import torch
from transformers import (
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
    pipeline,
    AutomaticSpeechRecognitionPipeline,
)

logger = getLogger(__name__)


def create_whisper_pipe() -> AutomaticSpeechRecognitionPipeline:
    device = "cuda"
    torch_dtype = torch.float16

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True,
        attn_implementation="flash_attention_2",
    )
    model.to(device)

    processor = AutoProcessor.from_pretrained(model_id)

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch_dtype,
        device=device,
    )

    return pipe


class WhisperOutput(TypedDict):
    text: str
    chunks: list["Chunk"]


class Chunk(TypedDict):
    timestamp: tuple[float, float]
    text: str


model_id = "openai/whisper-large-v3-turbo"


@dataclass
class Whisper:
    model_id = model_id

    pipe: AutomaticSpeechRecognitionPipeline = field(
        default_factory=create_whisper_pipe
    )

    def transcribe_file(self, file: str | Path, translate=False) -> str:
        if isinstance(file, Path):
            file = str(file)
        assert isinstance(file, str)
        logger.info("Transcribing file: %s", file)
        try:
            output: WhisperOutput = self.run_pipe(
                file, translate, return_timestamps=False
            )
            return text(output)
        except ValueError as e:
            if "Please either pass `return_timestamps=True`" in repr(e):
                logger.info("File is >30s, transcribing with timestamps: %s", file)
                output = self.run_pipe(file, translate, return_timestamps=True)
                return text(output)
            raise

    def run_pipe(
        self, file: str, translate: bool, return_timestamps: bool
    ) -> WhisperOutput:
        kwargs: dict[str, Any] = {"return_timestamps": return_timestamps}
        if translate:
            kwargs["generate_kwargs"] = {"language": "english"}
        # ignore this warning:
        # .../site-packages/transformers/models/whisper/generation_whisper.py:496: FutureWarning: The input name `inputs` is deprecated. Please make sure to use `input_features` instead.
        with warnings.catch_warnings(action="ignore", category=FutureWarning):
            data: WhisperOutput = self.pipe(file, **kwargs)
        return data


def text(data: WhisperOutput) -> str:
    return data["text"].strip()


def merge_prompt_with_transcription(prompt: str, data: WhisperOutput) -> str:
    pretty_json = json.dumps(data["chunks"], indent=2, ensure_ascii=False)
    merged: str = f"""{prompt}

Additional context:
The following is the chunked transcription by the Whisper model:
```json
{pretty_json}
```
"""
    return merged
