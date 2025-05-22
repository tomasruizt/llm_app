from dataclasses import dataclass, field
import logging
import os
from typing import Iterable, AsyncGenerator
import requests
import aiohttp
import asyncio
from ..base_llm import LLM, Conversation, Message
from ..rest_api.restapi_client import encode_as_png_in_base64
from multiprocessing import Pool

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
    payload_kwargs: dict = field(default_factory=dict)
    remote_call_concurrency: int = 8

    def headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    model_ids = [_default_model, "gpt-4o"]

    def complete(self, prompt: str) -> str:
        return complete(model=self, prompt=prompt)

    def complete_many(
        self, prompts: list[str], n_workers: int = os.cpu_count()
    ) -> list[str]:
        return complete_many(model=self, prompts=prompts, n_workers=n_workers)

    def complete_msgs(
        self, msgs: list[Message], output_dict: bool = False
    ) -> str | dict:
        messages: list[dict] = extract_msgs(msgs)
        completion: dict = complete_msgs(model=self, messages=messages)
        data: dict = as_dict(completion)
        if not output_dict:
            return data["response"]
        return data

    def complete_batch(self, batch: Iterable[Conversation]) -> Iterable[dict]:
        """Process a batch of conversations asynchronously.

        Args:
            batch: An iterable of Conversation objects

        Returns:
            An iterable of dictionaries containing the completion results
        """
        listof_convos = (extract_msgs(convo) for convo in batch)

        params = {"model": self.model_id, "temperature": 0.0, **self.payload_kwargs}

        agen = _batch_call_openai(
            base_url=self.base_url,
            headers=self.headers(),
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


def as_dict(completion: dict) -> dict:
    message = completion["choices"][0]["message"]
    data = {
        "response": message["content"],
        "n_input_tokens": completion["usage"]["prompt_tokens"],
        "n_output_tokens": completion["usage"]["completion_tokens"],
    }
    if "reasoning" in message:
        data["reasoning"] = message["reasoning"]
    return data


def complete_many(
    model: OpenAIModel, prompts: list[str], n_workers: int = os.cpu_count()
) -> list[str]:
    print("Calling OpenAI API")
    with Pool(processes=n_workers) as pool:
        args = [(model, p) for p in prompts]
        return pool.starmap(complete, args)


def complete(model: OpenAIModel, prompt: str) -> str:
    messages = [{"role": "user", "content": prompt}]
    return complete_msgs(model=model, messages=messages)


def complete_msgs(model: OpenAIModel, messages: list[dict]) -> dict:
    url = f"{model.base_url}/chat/completions"
    payload = {
        "model": model.model_id,
        "temperature": 0.0,
        "messages": messages,
        **model.payload_kwargs,
    }

    response = requests.post(url, headers=model.headers(), json=payload)
    response.raise_for_status()
    completion: dict = response.json()
    return completion


def postprocess(response: str) -> str:
    return response.lower().strip(".").strip()


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
        "payload_kwargs": {"provider": {"only": ["Cerebras"]}},
    }


async def _batch_call_openai(
    base_url: str,
    headers: dict,
    iterof_messages: Iterable[list[dict]],
    params: dict,
    n_concurrency: int,
) -> AsyncGenerator[dict, None]:
    tasks = []
    semaphore = asyncio.Semaphore(n_concurrency)

    async with aiohttp.ClientSession() as session:
        for request_idx, messages in enumerate(iterof_messages):
            coro = _call_openai(
                session=session,
                base_url=base_url,
                headers=headers,
                messages=messages,
                params=params,
                request_idx=request_idx,
                semaphore=semaphore,
            )
            tasks.append(coro)

        for task in asyncio.as_completed(tasks):
            yield await task


async def _call_openai(
    session: aiohttp.ClientSession,
    base_url: str,
    headers: dict,
    messages: list[dict],
    params: dict,
    request_idx: int,
    semaphore: asyncio.Semaphore,
) -> dict:
    async with semaphore:
        logger.info("Calling OpenAI API for request %d", request_idx)
        try:
            url = f"{base_url}/chat/completions"
            payload = {**params, "messages": messages}

            async with session.post(url, headers=headers, json=payload) as response:
                response.raise_for_status()
                completion = await response.json()

                asdict = as_dict(completion)
                asdict["request_idx"] = request_idx
                asdict["success"] = True
                return asdict

        except Exception as e:
            logger.error(
                "Error calling OpenAI API for request %d. Cause: %s",
                request_idx,
                repr(e),
            )
            return {"request_idx": request_idx, "error": str(e), "success": False}
