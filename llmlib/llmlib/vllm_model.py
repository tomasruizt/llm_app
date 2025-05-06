from pathlib import Path
import time
from typing import Any, Literal
from PIL import Image
from vllm.engine.arg_utils import EngineArgs
from transformers import AutoProcessor
from vllm import SamplingParams, LLM
from dataclasses import asdict, dataclass
from llmlib.base_llm import LLM as BaseLLM, Conversation, Message
from llmlib.huggingface_inference import is_img, video_to_imgs, is_video
import logging

logger = logging.getLogger(__name__)


@dataclass
class ModelvLLM(BaseLLM):
    """Inspired by https://github.com/vllm-project/vllm/blob/main/examples/offline_inference/vision_language_multi_image.py"""

    model_id: str  # e.g "google/gemma-3-4b-it"
    max_n_frames_per_video: int = 100
    max_new_tokens: int = 500
    gpu_size: Literal["24GB", "80GB"] = "24GB"
    max_model_len: int = 8192

    def __post_init__(self):
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self._llm = None

    def get_llm(self) -> LLM:
        """Lazy initialization of the LLM to get quicker feedback on unit tests."""
        if self._llm is not None:
            return self._llm

        batch_size = 8 if self.gpu_size == "80GB" else 2
        engine_args = EngineArgs(
            model=self.model_id,
            task="generate",
            max_model_len=self.max_model_len,
            max_num_batched_tokens=self.max_model_len * batch_size,
            max_num_seqs=batch_size,
            limit_mm_per_prompt={"image": self.max_n_frames_per_video},
            dtype="bfloat16",
        )
        engine_args = asdict(engine_args)  # type: ignore
        start = time.time()
        self._llm = LLM(**engine_args)  # type: ignore
        logger.info("Time spent in vLLM LLM() constructor: %s", time.time() - start)
        return self.get_llm()

    def complete_msgs(self, msgs: list[Message], **generate_kwargs) -> str:
        return self.complete_batch([msgs], **generate_kwargs)[0]

    def complete_batch(
        self, batch: list[Conversation], output_dict: bool = False, **generate_kwargs
    ) -> list[str]:
        assert all(len(convo) == 1 for convo in batch), (
            "Each convo must have exactly one message"
        )

        n_frames_per_convo: list[int] = []
        listof_inputs: list[dict[str, Any]] = []
        for convo in batch:
            inputs, n_frames = to_vllm_format(
                self.processor,
                message=convo[0],
                max_n_frames_per_video=self.max_n_frames_per_video,
            )
            listof_inputs.append(inputs)
            n_frames_per_convo.append(n_frames)

        params = dict(temperature=1.0, max_tokens=self.max_new_tokens) | generate_kwargs

        start = time.time()
        outputs = self.get_llm().generate(
            listof_inputs,  # type: ignore
            sampling_params=SamplingParams(**params),
        )
        runtime = time.time() - start

        responses: list[str] = [o.outputs[0].text for o in outputs]
        if not output_dict:
            return responses

        data = {
            "request_id": [o.request_id for o in outputs],
            "response": responses,
            "n_input_tokens": [len(o.prompt_token_ids) for o in outputs],  # type: ignore
            "n_output_tokens": [len(o.outputs[0].token_ids) for o in outputs],
            "n_frames": n_frames_per_convo,
            "model_runtime": [runtime / len(batch)] * len(batch),  # average runtime
        }
        return data


def to_vllm_format(
    processor: AutoProcessor, message: Message, max_n_frames_per_video: int
) -> tuple[dict, int]:
    question = message.msg
    imgs = convert_media_to_listof_imgs(message, max_n_frames_per_video)
    n_frames = len(imgs)

    placeholders = [
        {"type": "image", "image": f"{idx}.jpeg"} for idx in range(n_frames)
    ]
    messages = [
        {"role": "user", "content": [*placeholders, {"type": "text", "text": question}]}
    ]

    prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    dict_input = {
        "prompt": prompt,
        "multi_modal_data": {"image": imgs},
    }
    return dict_input, n_frames


def convert_media_to_listof_imgs(
    msg: Message, max_n_frames_per_video: int
) -> list[Image.Image]:
    imgs: list[Image.Image] = []
    if msg.img is not None:
        if isinstance(msg.img, (str, Path)):
            imgs.append(Image.open(msg.img))
        else:
            imgs.append(msg.img)

    if msg.video is not None:
        frames = video_to_imgs(msg.video, max_n_frames=max_n_frames_per_video)
        imgs.extend(frames)

    if msg.files is not None:
        for filepath in msg.files:
            if is_img(filepath):
                imgs.append(Image.open(filepath))
            elif is_video(filepath):
                frames = video_to_imgs(filepath, max_n_frames=max_n_frames_per_video)
                imgs.extend(frames)
            else:
                raise ValueError(f"Unsupported file type: {filepath}")
    return imgs
