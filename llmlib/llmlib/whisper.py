from dataclasses import dataclass
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


def create_whisper_pipe(
    attn_implementation: str = "flash_attention_2",
    compile: bool = False,
) -> AutomaticSpeechRecognitionPipeline:
    device = "cuda"
    torch_dtype = torch.float16

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True,
        attn_implementation=attn_implementation,
    )
    model.to(device)
    if compile:
        logger.info("Compiling %s", model_id)
        model = torch.compile(model)

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
    use_flash_attention: bool = True
    compile: bool = False
    batch_size: int = 50

    def __post_init__(self):
        if self.use_flash_attention:
            attn_implementation = "flash_attention_2"
        else:
            attn_implementation = "sdpa"

        self.pipe = create_whisper_pipe(
            attn_implementation=attn_implementation,
            compile=self.compile,
        )

    def transcribe_file(self, file: str | Path, translate=False) -> str:
        if isinstance(file, Path):
            file = str(file)
        assert isinstance(file, str)
        logger.info("Transcribing file: %s", file)
        output: WhisperOutput = self.run_pipe(file, translate)
        return text(output)

    def transcribe_batch(
        self, files: list[str | Path], translate=False
    ) -> list[str | Exception]:
        """Transcribe multiple files in a batch for better GPU efficiency"""
        if len(files) == 0:
            return []

        file_paths = [str(f) if isinstance(f, Path) else f for f in files]
        logger.info("Transcribing batch of %d files", len(file_paths))

        try:
            # Try batch processing first for efficiency
            outputs = self.run_pipe(file_paths, translate)
            return [text(output) for output in outputs]
        except Exception as e:
            # If batch processing fails, fall back to individual processing
            logger.error(
                "Batch processing failed, falling back to individual processing: %s",
                repr(e),
            )
            results = []
            for file_path in file_paths:
                try:
                    output = self.run_pipe(file_path, translate)
                    results.append(text(output))
                except Exception as file_error:
                    results.append(file_error)
            return results

    def run_pipe(
        self, file_or_files: str | list[str], translate: bool
    ) -> WhisperOutput | list[WhisperOutput]:
        """Run the pipeline on a single file or a list of files"""
        kwargs: dict[str, Any] = {"return_timestamps": True}
        if translate:
            kwargs["generate_kwargs"] = {"language": "english"}
        # ignore this warning:
        # .../site-packages/transformers/models/whisper/generation_whisper.py:496: FutureWarning: The input name `inputs` is deprecated. Please make sure to use `input_features` instead.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            data = self.pipe(
                file_or_files,
                **kwargs,
                batch_size=self.batch_size,
                chunk_length_s=30,
            )
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
