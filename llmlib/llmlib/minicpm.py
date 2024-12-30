from io import BytesIO
import logging
from pathlib import Path
from typing import Any
from llmlib.base_llm import LLM, Message
import torch
from transformers import AutoModel, AutoTokenizer
from PIL import Image
from decord import VideoReader, cpu  # pip install decord

logger = logging.getLogger(__name__)


_model_name = "openbmb/MiniCPM-V-2_6"


class MiniCPM(LLM):
    temperature: float

    model_id = _model_name
    requires_gpu_exclusively = True

    def __init__(self, temperature: float = 0.0, model=None) -> None:
        if model is None:
            model = _create_model()
        self.model = model
        self.tokenizer = _create_tokenizer()
        self.temperature = temperature

    def chat(self, prompt: str) -> str:
        return self.complete_msgs([Message(role="user", msg=prompt)])

    def complete_msgs(self, msgs: list[Message]) -> str:
        dict_msgs = [_convert_msg_to_dict(m) for m in msgs]
        use_sampling = self.temperature > 0.0
        res = self.model.chat(
            image=None,
            msgs=dict_msgs,
            tokenizer=self.tokenizer,
            sampling=use_sampling,
            temperature=self.temperature,
        )
        return res

    def video_prompt(self, video: Path | BytesIO, prompt: str) -> str:
        return video_prompt(self, video, prompt)


def _create_tokenizer():
    return AutoTokenizer.from_pretrained(_model_name, trust_remote_code=True)


def _create_model():
    model = AutoModel.from_pretrained(
        _model_name,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
    )
    model.eval().cuda()
    return model


def _convert_msg_to_dict(msg: Message) -> dict:
    if msg.img is None:
        content: list[Any] = [msg.msg]
    else:
        content = [msg.img.convert("RGB"), msg.msg]
    return {"role": msg.role, "content": content}


def to_listof_imgs(video: Path | BytesIO) -> list[Image.Image]:
    """
    Return one frame per second from the video.
    If the video is longer than MAX_NUM_FRAMES, sample MAX_NUM_FRAMES frames.
    """
    MAX_NUM_FRAMES = 64  # if cuda OOM set a smaller number
    if isinstance(video, Path):
        assert video.exists(), video
        vr = VideoReader(str(video), ctx=cpu(0))
    else:
        vr = VideoReader(BytesIO(video.read()), ctx=cpu(0))
    sample_fps = round(vr.get_avg_fps() / 1)  # FPS
    frame_idx = [i for i in range(0, len(vr), sample_fps)]
    if len(frame_idx) > MAX_NUM_FRAMES:
        frame_idx = uniform_sample(frame_idx, MAX_NUM_FRAMES)
    imgs = vr.get_batch(frame_idx).asnumpy()
    imgs = [Image.fromarray(v.astype("uint8")) for v in imgs]
    return imgs


def uniform_sample(xs, n):
    gap = len(xs) / n
    idxs = [int(i * gap + gap / 2) for i in range(n)]
    return [xs[i] for i in idxs]


def video_prompt(self: MiniCPM, video: Path | BytesIO, prompt: str) -> str:
    imgs = to_listof_imgs(video)
    logger.info("Video turned into %d images", len(imgs))
    msgs = [
        {"role": "user", "content": [prompt] + imgs},
    ]
    # Set decode params for video
    params = {}
    params["use_image_id"] = False
    params["max_slice_nums"] = 2  # use 1 if cuda OOM and video resolution >  448*448
    answer = self.model.chat(image=None, msgs=msgs, tokenizer=self.tokenizer, **params)
    return answer
