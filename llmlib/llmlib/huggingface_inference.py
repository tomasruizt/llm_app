import os
import base64
import io
from pathlib import Path
from dataclasses import dataclass
from huggingface_hub import InferenceClient
import PIL
from enum import StrEnum
from .base_llm import LLM, Message, validate_only_first_message_has_files


def get_image_as_base64(image_bytes: bytes):
    return base64.b64encode(image_bytes).decode("utf-8")


def convert_message_to_hf_format(message: Message) -> dict:
    """Convert a Message to HuggingFace chat format."""
    content = []

    # Add text content if present
    if message.msg:
        content.append({"type": "text", "text": message.msg})

    # Add single image content if present
    if message.img is not None:
        content.append(extract_content_piece(message.img))

    # Add multiple images from files if present
    if message.files is not None:
        for file_path in message.files:
            if is_img(file_path):
                content.append(extract_content_piece(file_path))

    return {"role": message.role, "content": content}


def is_img(file_path: str | Path) -> bool:
    permitted = (".png", ".jpg", ".jpeg")
    return str(file_path).lower().endswith(permitted)


def extract_content_piece(img: PIL.Image.Image | str | Path) -> dict:
    image_bytes = extract_bytes(img)
    content_piece = {
        "type": "image_url",
        "image_url": {
            "url": f"data:image/jpeg;base64,{get_image_as_base64(image_bytes)}"
        },
    }
    return content_piece


def extract_bytes(img: PIL.Image.Image | str | Path) -> bytes:
    if isinstance(img, (str, Path)):
        # Handle file path
        with open(img, "rb") as f:
            return f.read()
    elif isinstance(img, PIL.Image.Image):
        # Handle PIL Image
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format=img.format or "PNG")
        return img_byte_arr.getvalue()
    else:
        raise ValueError(f"Unsupported image type: {type(img)}")


class HuggingFaceVLMs(StrEnum):
    gemma_3_27b_it = "google/gemma-3-27b-it"


@dataclass
class HuggingFaceVLM(LLM):
    """Base class for HuggingFace Vision Language Models."""

    model_id: HuggingFaceVLMs
    max_new_tokens: int = 1000
    requires_gpu_exclusively: bool = False

    # Available model IDs
    model_ids = list(HuggingFaceVLMs)

    def __post_init__(self):
        """Initialize the HuggingFace client after dataclass initialization."""
        if "HF_TOKEN_INFERENCE" not in os.environ:
            raise ValueError("HF_TOKEN_INFERENCE environment variable is required")

        self.client = InferenceClient(
            provider="hf-inference",
            api_key=os.environ["HF_TOKEN_INFERENCE"],
        )

    def complete_msgs(self, msgs: list[Message]) -> str:
        """Complete a conversation with the model."""
        validate_only_first_message_has_files(msgs)
        hf_messages = [convert_message_to_hf_format(msg) for msg in msgs]

        completion = self.client.chat.completions.create(
            model=self.model_id,
            messages=hf_messages,
            max_tokens=self.max_new_tokens,
        )

        return completion.choices[0].message.content

    @staticmethod
    def get_info() -> list[str]:
        """Get information about the model."""
        return [
            "Uses HuggingFace Inference API (huggingface.co)",
            "Supports text-only and image+text queries",
            "Supports multi-turn conversations",
        ]
