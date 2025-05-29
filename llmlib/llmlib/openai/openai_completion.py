from dataclasses import dataclass, field
from itertools import cycle
import logging
import os
from typing import Generator, Iterable, AsyncGenerator
import aiohttp
import asyncio
from ..base_llm import LLM, Conversation, Message
from ..rest_api.restapi_client import encode_as_png_in_base64

logger = logging.getLogger(__name__)

_default_model = "gpt-4o-mini"


def get_openai_api_key() -> str:
    logger.info("Reading OpenAI API key from environment variable")
    return os.environ["OPENAI_API_KEY"]


@dataclass
class OpenAIModel(LLM):
    model_id: str = _default_model
    base_url: str = "https://api.openai.com/v1"
    api_key: str = field(default_factory=get_openai_api_key)
    generation_kwargs: dict = field(default_factory=dict)
    remote_call_concurrency: int = 32

    def headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    model_ids = [_default_model, "gpt-4o"]

    def complete_msgs(
        self, msgs: Conversation, output_dict: bool = False
    ) -> str | dict:
        for data in self.complete_batch([msgs]):
            pass  # avoid RuntimeError: async generator ignored GeneratorExit
        if not output_dict:
            return data["response"]
        return data

    def complete_batch(
        self, batch: Iterable[Conversation], metadatas: Iterable[dict] | None = None
    ) -> Iterable[dict]:
        listof_convos = (extract_msgs(convo) for convo in batch)

        gen_kwargs = {"model": self.model_id, "temperature": 0.0}
        gen_kwargs = gen_kwargs | self.generation_kwargs

        agen: AsyncGenerator[dict, None] = _batch_call_openai(
            base_url=self.base_url,
            headers=self.headers(),
            iterof_messages=listof_convos,
            metadatas=metadatas,
            remote_call_concurrency=self.remote_call_concurrency,
            gen_kwargs=gen_kwargs,
        )
        gen = to_synchronous_generator(agen)
        return gen


def to_synchronous_generator(
    agen: AsyncGenerator[dict, None],
) -> Generator[dict, None, None]:
    loop = asyncio.get_event_loop()
    try:
        while True:
            output: dict = loop.run_until_complete(agen.__anext__())
            yield output
    except StopAsyncIteration:
        pass


def as_dict(completion: dict) -> dict:
    message = completion["choices"][0]["message"]
    data = {
        "response": message["content"],
        "n_input_tokens": completion["usage"]["prompt_tokens"],
        "n_output_tokens": completion["usage"]["completion_tokens"],
    }
    if "reasoning" in message:  # OpenAI format
        data["reasoning"] = message["reasoning"]
    elif "reasoning_content" in message:  # vLLM format
        data["reasoning"] = message["reasoning_content"]
    return data


def extract_msgs(msgs: list[Message]) -> list[dict]:
    return [extract_msg(m) for m in msgs]


def extract_msg(msg: Message) -> dict:
    if msg.img is None:
        return {"role": msg.role, "content": msg.msg}
    img_in_base64 = encode_as_png_in_base64(msg.img)
    return {
        "role": msg.role,
        "content": [
            {"type": "text", "text": msg.msg},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{img_in_base64}"},
            },
        ],
    }


def config_for_cerebras_on_openrouter() -> dict:
    """kwargs for OpenAIModel to use Cerebras on OpenRouter"""
    logger.info("Reading OpenRouter API key from environment variable")
    return {
        "base_url": "https://openrouter.ai/api/v1",
        "api_key": os.environ["OPENROUTER_API_KEY"],
        "generation_kwargs": {"provider": {"only": ["Cerebras"]}},
    }


async def _batch_call_openai(
    base_url: str,
    headers: dict,
    iterof_messages: Iterable[list[dict]],
    gen_kwargs: dict,
    remote_call_concurrency: int,
    timeout_secs: int = 60,
    metadatas: Iterable[dict] | None = None,
) -> AsyncGenerator[dict, None]:
    if metadatas is None:
        metadatas = cycle([{}])

    tasks = []
    timeout = aiohttp.ClientTimeout(sock_connect=timeout_secs, sock_read=timeout_secs)
    connector = aiohttp.TCPConnector(limit=remote_call_concurrency)
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        for request_idx, (messages, metadata) in enumerate(
            zip(iterof_messages, metadatas)
        ):
            post_kwargs = {
                "url": f"{base_url}/chat/completions",
                "headers": headers,
                "json": {**gen_kwargs, "messages": messages},
            }
            metadata = metadata | {"request_idx": request_idx}
            coro = _call_openai(
                session=session,
                post_kwargs=post_kwargs,
                metadata=metadata,
            )
            tasks.append(coro)

        for task in asyncio.as_completed(tasks):
            yield await task


async def _call_openai(
    session: aiohttp.ClientSession,
    post_kwargs: dict,
    metadata: dict,
) -> dict:
    request_idx = metadata["request_idx"]
    logger.debug("Calling OpenAI API for request %d", request_idx)
    try:
        async with session.post(**post_kwargs) as response:
            response.raise_for_status()
            completion = await response.json()
        asdict = as_dict(completion) | metadata
        asdict["success"] = True
        return asdict

    except Exception as e:
        logger.error(
            "Error calling OpenAI API for request %d. Cause: %s",
            request_idx,
            repr(e),
        )
        return {"request_idx": request_idx, "error": str(e), "success": False}
