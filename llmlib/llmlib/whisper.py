from dataclasses import dataclass, field
from logging import getLogger
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

    def transcribe_file(self, file: str) -> str:
        assert isinstance(file, str)
        logger.info("Transcribing file: %s", file)
        return self.pipe(file)["text"].strip()
