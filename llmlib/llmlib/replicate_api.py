from io import BytesIO
from logging import getLogger
from pathlib import Path
from llmlib.base_llm import LLM
import replicate

logger = getLogger(__name__)


class Apollo7B(LLM):
    model_id = "replicate/lucataco/apollo-7b"
    model_id_full = "lucataco/apollo-7b:e282f76d0451b759128be3e8bccfe5ded8f521f4a7d705883e92f837e563f575"

    def video_prompt(self, video: Path | BytesIO, prompt: str) -> str:
        logger.info("Calling Replicate API with model %s", self.model_id)
        output = replicate.run(
            self.model_id_full,
            input={
                "top_p": 0.7,
                "video": video,
                "prompt": prompt,
                "temperature": 0.4,
                "max_new_tokens": 512,
            },
        )
        return output

    @classmethod
    def get_warnings(cls) -> list[str]:
        return [
            "This model only supports single-turn video chat.",
            "This model is run using the paid Replicate API ([see link](https://replicate.com/lucataco/apollo-7b)). The first call to it might take 30s while the model warms up.",
        ]
