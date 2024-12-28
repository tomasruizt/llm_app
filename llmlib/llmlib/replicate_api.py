from io import BytesIO
import json
from logging import getLogger
from pathlib import Path
from typing import Literal, TypedDict
from llmlib.base_llm import LLM, Message
import replicate

logger = getLogger(__name__)


class Msg(TypedDict):
    role: Literal["user", "assistant"]
    content: str


class Apollo7B(LLM):
    model_id = "replicate/lucataco/apollo-7b"
    model_id_full = "tomasruizt/apollo-7b-multiturn:1120d3a705916a77c094d3bfc180089f6a4e46f840218bd1663b00b889da2951"

    def video_prompt(self, video: Path | BytesIO, prompt: str) -> str:
        logger.info("Calling Replicate API with model %s", self.model_id)
        messages = [{"role": "user", "content": prompt}]
        return self.call_api(video=video, messages=messages)

    def complete_msgs2(self, msgs: list[Message]) -> str:
        fst_msg = msgs[0]
        if not fst_msg.has_video():
            raise ValueError("Apollo7B only supports video input")
        messages = [Msg(role=msg.role, content=msg.msg) for msg in msgs]
        return self.call_api(video=fst_msg.video, messages=messages)

    def call_api(self, video: Path | BytesIO, messages: list[Msg]) -> str:
        output = replicate.run(
            self.model_id_full,
            input={
                "top_p": 0.7,
                "video": video,
                "messages": json.dumps(messages),
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
