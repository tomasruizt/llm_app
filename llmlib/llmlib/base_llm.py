from pathlib import Path
from typing import Literal, Self
from PIL import Image


from dataclasses import dataclass


@dataclass
class Message:
    role: Literal["user", "assistant"]
    msg: str
    img_name: str | None = None
    img: Image.Image | None = None

    @classmethod
    def from_prompt(cls, prompt: str) -> Self:
        return cls(role="user", msg=prompt)


class LLM:
    model_id: str
    requires_gpu_exclusively: bool = False

    def complete_msgs2(self, msgs: list[Message]) -> str:
        raise NotImplementedError

    def complete_batch(self, batch: list[list[Message]]) -> list[str]:
        raise NotImplementedError

    def video_prompt(self, video_path: Path, prompt: str) -> str:
        raise NotImplementedError

    @classmethod
    def get_warnings(cls) -> list[str]:
        return []
