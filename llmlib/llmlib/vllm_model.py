from pathlib import Path
import time
from typing import Any, Iterable, AsyncGenerator
from openai import AsyncOpenAI
from openai.types.chat.chat_completion import ChatCompletion
import pandas as pd
from dataclasses import dataclass
from llmlib.base_llm import LLM as BaseLLM, Conversation
from llmlib.huggingface_inference import is_img, is_video
import logging
import asyncio

logger = logging.getLogger(__name__)


@dataclass
class ModelvLLM(BaseLLM):
    """OpenAI-compatible vLLM server: https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html"""

    model_id: str  # e.g "google/gemma-3-4b-it"
    max_new_tokens: int = 500
    temperature: float = 0.0
    remote_call_concurrency: int = 8
    port: int = 8000

    def complete_msgs(
        self, msgs: Conversation, output_dict: bool = False, **generate_kwargs
    ) -> str | dict:
        gen = self.complete_batch([msgs], **generate_kwargs)
        result: dict = next(gen)
        if output_dict:
            return result
        else:
            return result["response"]

    def complete_batch(
        self,
        batch: Iterable[Conversation],
        **generate_kwargs,
    ) -> Iterable[dict]:
        listof_convos = (to_vllm_oai_format(convo) for convo in batch)

        params = dict(
            model=self.model_id,
            temperature=self.temperature,
            max_tokens=self.max_new_tokens,
        )
        params |= generate_kwargs

        server = f"http://localhost:{self.port}/v1"
        client = AsyncOpenAI(api_key="EMPTY", base_url=server)
        agen = _batch_call_vllm_server(
            client=client,
            iterof_messages=listof_convos,
            params=params,
            n_concurrency=self.remote_call_concurrency,
        )
        loop = asyncio.get_event_loop()
        try:
            while True:
                output: dict = loop.run_until_complete(agen.__anext__())
                yield output
        except StopAsyncIteration:
            pass


def as_completion_dict(c: ChatCompletion) -> dict:
    return {
        "response": c.choices[0].message.content,
        "n_input_tokens": c.usage.prompt_tokens,
        "n_output_tokens": c.usage.completion_tokens,
    }


async def _batch_call_vllm_server(
    client: AsyncOpenAI,
    iterof_messages: Iterable[list[dict]],
    params: dict,
    n_concurrency: int,
) -> AsyncGenerator[dict, None]:
    tasks = []
    semaphore = asyncio.Semaphore(n_concurrency)
    for request_idx, messages in enumerate(iterof_messages):
        coro = _call_vllm_server(client, messages, params, request_idx, semaphore)
        tasks.append(coro)

    for task in asyncio.as_completed(tasks):
        yield await task


async def _call_vllm_server(
    client: AsyncOpenAI,
    messages: list[dict],
    params: dict,
    request_idx: int,
    semaphore: asyncio.Semaphore,
) -> dict:
    async with semaphore:
        logger.info("Calling vLLM server for request %d", request_idx)
        try:
            start = time.time()
            completion: ChatCompletion = await client.chat.completions.create(
                messages=messages, **params
            )
            runtime = time.time() - start
        except Exception as e:
            # Error path
            logger.error(
                "Error calling vLLM server for request %d. Cause: %s",
                request_idx,
                repr(e),
            )
            asdict = {"request_idx": request_idx, "error": e, "success": False}
            return asdict

    # Happy path
    asdict: dict = as_completion_dict(completion)
    asdict["request_idx"] = request_idx
    asdict["success"] = True
    asdict["model_runtime"] = runtime
    return asdict


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
