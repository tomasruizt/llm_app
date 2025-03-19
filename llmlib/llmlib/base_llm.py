from io import BytesIO
from pathlib import Path
from typing import Literal
from typing_extensions import Self
from PIL import Image


from dataclasses import dataclass


@dataclass
class Message:
    role: Literal["user", "assistant"]
    msg: str
    img_name: str | None = None
    img: Path | Image.Image | None = None
    video: Path | BytesIO | None = None
    # TODO: make default files an empty list
    files: list[Path] | None = None

    @classmethod
    def from_prompt(cls, prompt: str) -> Self:
        return cls(role="user", msg=prompt)

    def has_video(self) -> bool:
        return self.video is not None

    def has_image(self) -> bool:
        return self.img is not None


class LLM:
    model_id: str
    requires_gpu_exclusively: bool = False

    def complete_msgs(self, msgs: list[Message]) -> str:
        raise NotImplementedError

    def complete_batch(self, batch: list[list[Message]]) -> list[str]:
        raise NotImplementedError

    def video_prompt(self, video: Path | BytesIO, prompt: str) -> str:
        raise NotImplementedError

    @classmethod
    def get_warnings(cls) -> list[str]:
        return []

    @classmethod
    def get_info(cls) -> list[str]:
        return []


def validate_only_first_message_has_files(messages: list[Message]) -> None:
    """Validate that only the first message can have file(s)."""
    for msg in messages[1:]:
        if msg.has_image() or msg.has_video() or msg.files is not None:
            raise ValueError("Only the first message can have file(s)")
