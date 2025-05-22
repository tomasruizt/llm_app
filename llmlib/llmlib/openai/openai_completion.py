from dataclasses import dataclass, field
import logging
import os
import requests
from ..base_llm import LLM, Message
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
