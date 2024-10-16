from dataclasses import dataclass, field
from logging import getLogger
from typing import Any
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


model_id = "openai/whisper-large-v3-turbo"


@dataclass
class Whisper:
    model_id = model_id

    pipe: AutomaticSpeechRecognitionPipeline = field(
        default_factory=create_whisper_pipe
    )

    def transcribe_file(self, file: str, translate=False) -> str:
        assert isinstance(file, str)
        logger.info("Transcribing file: %s", file)
        try:
            return self._transcribe(file, translate, return_timestamps=False)
        except ValueError as e:
            if "Please either pass `return_timestamps=True`" in repr(e):
                logger.info("File is >30s, transcribing with timestamps: %s", file)
                return self._transcribe(file, translate, return_timestamps=True)
            raise

    def _transcribe(self, file: str, translate: bool, return_timestamps: bool) -> str:
        kwargs: dict[str, Any] = {"return_timestamps": return_timestamps}
        if translate:
            kwargs["generate_kwargs"] = {"language": "english"}
        # ignore this warning:
        # .../site-packages/transformers/models/whisper/generation_whisper.py:496: FutureWarning: The input name `inputs` is deprecated. Please make sure to use `input_features` instead.
        with warnings.catch_warnings(action="ignore", category=FutureWarning):
            # data["chunks"] contains the timestamped transcriptions
            data = self.pipe(file, **kwargs)
        return data["text"].strip()
