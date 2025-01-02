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


class MiniCPM(LLM):
    temperature: float

    model_ids = ["openbmb/MiniCPM-V-2_6", "openbmb/MiniCPM-V-2_6-int4"]
    requires_gpu_exclusively = True

    def __init__(self, model_id: str, temperature: float = 0.0, model=None) -> None:
        if model is None:
            model = _create_model(model_id)
        self.model_id = model_id
        self.model = model
        self.tokenizer = _create_tokenizer(model_id)
        self.temperature = temperature

    def chat(self, prompt: str) -> str:
        return self.complete_msgs([Message(role="user", msg=prompt)])

    def complete_msgs(self, msgs: list[Message]) -> str:
        dict_msgs = [_convert_msg_to_dict(m) for m in msgs]
        res = self.model.chat(
            image=None,
            msgs=dict_msgs,
            tokenizer=self.tokenizer,
            **self._chat_kwargs(),
        )
        return res

    def _chat_kwargs(self) -> dict[str, Any]:
        return {
            "sampling": self.temperature > 0.0,
            "temperature": self.temperature,
        }

    def video_prompt(self, video: Path | BytesIO, prompt: str) -> str:
        return video_prompt(self, video, prompt)

    @classmethod
    def get_info(cls) -> list[str]:
        return [
            "MiniCPM-V 2.6 by OpenBMB. Hugging face link [here](https://huggingface.co/openbmb/MiniCPM-V-2_6)",
            "This model supports multi-turn conversations with an image or a video.",
        ]

    def is_quantized(self) -> bool:
        return self.model_id.endswith("-int4")


def _create_tokenizer(model_id: str):
    return AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)


def _create_model(model_id: str):
    model = AutoModel.from_pretrained(
        model_id,
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


def to_listof_imgs(video: Path | BytesIO, max_num_frames: int) -> list[Image.Image]:
    """
    Return one frame per second from the video.
    If the video is longer than max_num_frames, sample max_num_frames frames uniformly.
    """
    if isinstance(video, Path):
        assert video.exists(), video
        vr = VideoReader(str(video), ctx=cpu(0))
    else:
        vr = VideoReader(BytesIO(video.read()), ctx=cpu(0))
    sample_fps = round(vr.get_avg_fps() / 1)  # FPS
    frame_idx = [i for i in range(0, len(vr), sample_fps)]
    if len(frame_idx) > max_num_frames:
        frame_idx = uniform_sample(frame_idx, max_num_frames)
    imgs = vr.get_batch(frame_idx).asnumpy()
    imgs = [Image.fromarray(v.astype("uint8")) for v in imgs]
    return imgs


def uniform_sample(xs, n):
    gap = len(xs) / n
    idxs = [int(i * gap + gap / 2) for i in range(n)]
    return [xs[i] for i in idxs]


def video_prompt(self: MiniCPM, video: Path | BytesIO, prompt: str) -> str:
    # If CUDA OOM, set max_num_frames to a smaller number.
    max_num_frames = 128 if self.is_quantized() else 64
    max_input_len = 8192  # 8192 is the default in model.chat()
    if self.is_quantized():
        max_input_len = 2 * max_input_len

    imgs = to_listof_imgs(video, max_num_frames)
    logger.info("Video turned into %d images", len(imgs))
    msgs = [
        {"role": "user", "content": [prompt] + imgs},
    ]
    # Set decode params for video
    params = self._chat_kwargs()
    params["use_image_id"] = False
    params["max_slice_nums"] = 2  # use 1 if cuda OOM and video resolution >  448*448
    params["max_inp_length"] = max_input_len
    answer = self.model.chat(image=None, msgs=msgs, tokenizer=self.tokenizer, **params)
    return answer
