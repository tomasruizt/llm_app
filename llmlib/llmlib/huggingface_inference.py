import math
import os
import base64
import io
from pathlib import Path
from dataclasses import dataclass
import PIL
from enum import Enum
import openai
from .base_llm import LLM, Message, validate_only_first_message_has_files
import cv2
from PIL import Image
from logging import getLogger
from cachetools.func import ttl_cache

logger = getLogger(__name__)


def get_image_as_base64(image_bytes: bytes):
    return base64.b64encode(image_bytes).decode("utf-8")


def convert_message_to_openai_format(
    message: Message, max_n_frames_per_video: int
) -> dict:
    """
    Convert a Message to OpenAI chat format.
    Images become base64 encoded strings.
    Videos are processed like a list of images, each of which becomes a base64 encoded string.
    """
    content = []

    # Add text content if present
    if message.msg:
        content.append({"type": "text", "text": message.msg})

    # Add single image content if present
    if message.img is not None:
        content.append(extract_content_piece(message.img))

    if message.video is not None:
        imgs: list = video_to_imgs(message.video, max_n_frames_per_video)
        for frame in imgs:
            content.append(extract_content_piece(frame))

    # Add multiple images from files (img or video) if present
    if message.files is not None:
        for file_path in message.files:
            if is_img(file_path):
                content.append(extract_content_piece(file_path))
            elif is_video(file_path):
                imgs: list = video_to_imgs(file_path, max_n_frames_per_video)
                for frame in imgs:
                    content.append(extract_content_piece(frame))
            else:
                raise ValueError(f"Unsupported file type: {file_path}")

    return {"role": message.role, "content": content}


@ttl_cache(ttl=10 * 60)  # 10 minutes
def video_to_imgs(video_path: Path, max_n_frames: int) -> list[PIL.Image.Image]:
    """From https://github.com/agustoslu/simple-inference-benchmark/blob/5cec55787d34af65f0d11efc429c3d4de92f051a/utils.py#L79"""
    assert isinstance(video_path, Path), video_path
    assert video_path.exists(), video_path
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    frame_indices = compute_frame_indices(
        vid_n_frames=total_frames, vid_fps=fps, max_n_frames=max_n_frames
    )

    frames = []
    for frame_idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        success, frame = cap.read()
        if success:
            # Convert BGR (the default format for OpenCV) to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame_rgb))

    cap.release()
    logger.info(f"Extracted {len(frames)} frames from video {video_path}")
    return frames


def compute_frame_indices(vid_n_frames: int, vid_fps: float, max_n_frames: int):
    """
    From https://github.com/agustoslu/simple-inference-benchmark/blob/5cec55787d34af65f0d11efc429c3d4de92f051a/utils.py#L164
    This function will return the frames starting at 0 every second.
    Unless that number exceeds max_n_frames, in which case it will return max_n_frames frames evenly spaced out, starting at 0.
    """
    assert isinstance(vid_n_frames, int), vid_n_frames
    assert isinstance(max_n_frames, int), max_n_frames
    vid_fps = int(vid_fps)
    fps_n_frames = math.ceil(vid_n_frames / vid_fps)
    if fps_n_frames <= max_n_frames:
        return list(range(0, vid_n_frames - 1, vid_fps))
    else:
        return list(range(0, vid_n_frames - 1, vid_n_frames // max_n_frames))


def is_img(file_path: str | Path) -> bool:
    permitted = (".png", ".jpg", ".jpeg")
    return str(file_path).lower().endswith(permitted)


def is_video(file_path: str | Path) -> bool:
    permitted = (".mp4",)
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
        img.save(img_byte_arr, format="jpeg")
        return img_byte_arr.getvalue()
    else:
        raise ValueError(f"Unsupported image type: {type(img)}")


class HuggingFaceVLMs(str, Enum):
    gemma_3_27b_it = "google/gemma-3-27b-it"


urls = {
    "serverless": "https://router.huggingface.co/hf-inference/v1",
    "hosted": "https://d3zeqo83ufwxs1k3.us-east4.gcp.endpoints.huggingface.cloud/v1/",
}


@dataclass
class HuggingFaceVLM(LLM):
    """Base class for HuggingFace Vision Language Models."""

    model_id: HuggingFaceVLMs
    max_new_tokens: int = 1000
    requires_gpu_exclusively: bool = False
    max_n_frames_per_video: int = 200
    use_hosted_model: bool = False

    # Available model IDs
    model_ids = list(HuggingFaceVLMs)

    def __post_init__(self):
        """Initialize the HuggingFace client after dataclass initialization."""
        if "HF_TOKEN_INFERENCE" not in os.environ:
            raise ValueError("HF_TOKEN_INFERENCE environment variable is required")

        if self.use_hosted_model:
            base_url = urls["hosted"]
        else:
            base_url = urls["serverless"]

        self.client = openai.OpenAI(
            base_url=base_url,
            api_key=os.environ["HF_TOKEN_INFERENCE"],
        )

    def complete_msgs(self, msgs: list[Message]) -> str:
        """Complete a conversation with the model."""
        validate_only_first_message_has_files(msgs)
        hf_messages = [
            convert_message_to_openai_format(
                msg, max_n_frames_per_video=self.max_n_frames_per_video
            )
            for msg in msgs
        ]

        logger.info("Calling HuggingFace API...")
        completion = self.client.chat.completions.create(
            model=self.model_id,
            messages=hf_messages,
            max_tokens=self.max_new_tokens,
        )
        logger.info("Token usage: %s", dict(completion.usage))

        return completion.choices[0].message.content

    @staticmethod
    def get_info() -> list[str]:
        """Get information about the model."""
        return [
            "Uses HuggingFace Inference API (huggingface.co)",
            "Supports text-only and image+text queries",
            "Supports multi-turn conversations",
        ]
