from pathlib import Path
import time
from typing import Any, Literal, Iterable
from PIL import Image
import pandas as pd
from transformers import AutoProcessor
from vllm import AsyncEngineArgs, AsyncLLMEngine, SamplingParams, RequestOutput
from dataclasses import dataclass
from llmlib.base_llm import LLM as BaseLLM, Conversation, Message
from llmlib.huggingface_inference import is_img, video_to_imgs, is_video
import logging
import asyncio
from typing import AsyncGenerator

logger = logging.getLogger(__name__)


@dataclass
class ModelvLLM(BaseLLM):
    """Inspired by https://github.com/vllm-project/vllm/blob/main/examples/offline_inference/vision_language_multi_image.py"""

    model_id: str  # e.g "google/gemma-3-4b-it"
    max_n_frames_per_video: int = 100
    max_new_tokens: int = 500
    gpu_size: Literal["24GB", "80GB"] = "24GB"
    max_model_len: int = 8192
    enforce_eager: bool = False

    def __post_init__(self):
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self._llm = None

    def get_llm(self) -> AsyncLLMEngine:
        """Lazy initialization of the LLM to get quicker feedback on unit tests."""
        if self._llm is not None:
            return self._llm

        batch_size = 8 if self.gpu_size == "80GB" else 2
        engine_args = AsyncEngineArgs(
            model=self.model_id,
            task="generate",
            max_model_len=self.max_model_len,
            max_num_batched_tokens=self.max_model_len * batch_size,
            max_num_seqs=batch_size,
            limit_mm_per_prompt={"image": self.max_n_frames_per_video},
            dtype="bfloat16",
            enforce_eager=self.enforce_eager,
        )
        start = time.time()
        self._llm = AsyncLLMEngine.from_engine_args(engine_args)
        logger.info("Time spent in vLLM LLM() constructor: %s", time.time() - start)
        return self.get_llm()

    def complete_msgs(self, msgs: Conversation, **generate_kwargs) -> str:
        gen = self.complete_batch([msgs], **generate_kwargs)
        return next(gen)

    def complete_batch(
        self,
        batch: Iterable[Conversation],
        output_dict: bool = False,
        **generate_kwargs,
    ) -> Iterable[str | dict]:
        # Convert iterable to list for processing
        batch_list = list(batch)
        assert all(len(convo) == 1 for convo in batch_list), (
            "Each convo must have exactly one message"
        )

        n_frames_per_convo: list[int] = []
        listof_inputs: list[dict[str, Any]] = []
        for convo in batch_list:
            inputs, n_frames = to_vllm_format(
                self.processor,
                message=convo[0],
                max_n_frames_per_video=self.max_n_frames_per_video,
            )
            listof_inputs.append(inputs)
            n_frames_per_convo.append(n_frames)

        params = dict(temperature=1.0, max_tokens=self.max_new_tokens) | generate_kwargs
        engine = self.get_llm()

        loop = asyncio.get_event_loop()
        agen = _generate_batch_async(
            engine=engine,
            listof_inputs=listof_inputs,
            params=params,
            output_dict=output_dict,
            n_frames_per_convo=n_frames_per_convo,
        )
        try:
            while True:
                start = time.time()
                output: str | dict = loop.run_until_complete(agen.__anext__())
                runtime = time.time() - start
                if output_dict:
                    assert isinstance(output, dict)
                    output["model_runtime"] = runtime
                yield output
        except StopAsyncIteration:
            pass


def as_output_dict(output: RequestOutput) -> dict:
    return {
        "request_id": output.request_id,
        "response": output.outputs[0].text,
        "n_input_tokens": len(output.prompt_token_ids),  # type: ignore
        "n_output_tokens": len(output.outputs[0].token_ids),
    }


async def _generate_batch_async(
    engine: AsyncLLMEngine,
    listof_inputs: list[dict[str, Any]],
    params: dict,
    output_dict: bool,
    n_frames_per_convo: list[int],
) -> AsyncGenerator[str | dict, None]:
    tasks = []
    reqid_2_n_frames = {}
    for idx, (input, n_frames) in enumerate(zip(listof_inputs, n_frames_per_convo)):
        request_id = str(idx)
        coro = _generate_async(engine, input, params, request_id)
        tasks.append(coro)
        reqid_2_n_frames[request_id] = n_frames

    for task in asyncio.as_completed(tasks):
        output: RequestOutput = await task
        if output_dict:
            odict = as_output_dict(output)
            odict["n_frames"] = reqid_2_n_frames[odict["request_id"]]
            yield odict
        else:
            yield output.outputs[0].text


async def _generate_async(
    engine: AsyncLLMEngine, input: dict[str, Any], params: dict, request_id: str
) -> RequestOutput:
    gen = engine.generate(
        input,
        sampling_params=SamplingParams(**params),
        request_id=request_id,
    )
    async for output in gen:
        final_output = output
    return final_output


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

    dict_input = {"prompt": prompt}
    if len(imgs) > 0:
        dict_input["multi_modal_data"] = {"image": imgs}
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


def to_vllm_oai_format(convo: Conversation) -> list[dict]:
    """Convert a Conversation (list of Message objects) into a list of dictionaries
    that can be used to generate a response from vLLM.

    Args:
        convo: A list of Message objects representing a conversation

    Returns:
        A list of dictionaries with the following format:
        - For system messages: {"role": "system", "content": str}
        - For user messages: {"role": "user", "content": list} where content contains
          text and media items (images/videos) in the correct format
    """
    formatted_messages = []

    for msg in convo:
        content = []

        # Add text content if present
        if msg.msg:
            content.append({"type": "text", "text": msg.msg})

        # Add image if present
        if msg.img is not None:
            assert isinstance(msg.img, (str, Path)), (
                f"msg.img must be a string or Path, got {type(msg.img)}"
            )
            img_path = str(Path(msg.img).absolute())
            content.append(
                {"type": "image_url", "image_url": {"url": f"file://{img_path}"}}
            )

        # Add video if present
        if msg.video is not None:
            assert isinstance(msg.video, (str, Path)), (
                f"msg.video must be a string or Path, got {type(msg.video)}"
            )
            video_path = str(Path(msg.video).absolute())
            content.append(
                {"type": "video_url", "video_url": {"url": f"file://{video_path}"}}
            )

        # Add files if present
        if msg.files is not None:
            for file in msg.files:
                file_path = str(Path(file).absolute())
                if is_img(file):
                    content.append(
                        {
                            "type": "image_url",
                            "image_url": {"url": f"file://{file_path}"},
                        }
                    )
                elif is_video(file):
                    content.append(
                        {
                            "type": "video_url",
                            "video_url": {"url": f"file://{file_path}"},
                        }
                    )

        # For system messages, content is just the text
        if msg.role == "system":
            formatted_messages.append({"role": "system", "content": msg.msg})
        # For user messages, content is a list of text and media items
        else:
            formatted_messages.append({"role": msg.role, "content": content})

    return formatted_messages


def dump_dataset_as_batch_request(
    dataset: list[Conversation], model_id: str, tgt_jsonl: Path, **generation_kwargs
):
    entries = []
    for idx, convo in enumerate(dataset):
        oai_format = to_vllm_oai_format(convo)
        entry: dict = to_batch_entry(
            oai_format,
            model_id=model_id,
            custom_id=str(idx),
            **generation_kwargs,
        )
        entries.append(entry)
    df = pd.DataFrame(entries)

    tgt_jsonl.parent.mkdir(parents=True, exist_ok=True)
    if tgt_jsonl.exists():
        logger.info("Overwriting batch file '%s'", tgt_jsonl)
    df.to_json(tgt_jsonl, orient="records", lines=True)
    logger.info("Dumped %d entries to '%s'", len(entries), tgt_jsonl)


def to_batch_entry(
    messages: list[dict], model_id: str, custom_id: str, **generation_kwargs
) -> dict[str, Any]:
    body = {"messages": messages, "model": model_id, **generation_kwargs}
    return {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": body,
    }
