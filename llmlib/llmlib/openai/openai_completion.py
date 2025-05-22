from dataclasses import dataclass
import os
import requests
from ..base_llm import LLM, Message
from ..rest_api.restapi_client import encode_as_png_in_base64
from multiprocessing import Pool

_default_model = "gpt-4o-mini"


@dataclass
class OpenAIModel(LLM):
    model: str = _default_model
    base_url: str = "https://api.openai.com/v1"
    api_key: str = os.environ["OPENAI_API_KEY"]

    def headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    model_ids = [_default_model, "gpt-4o"]

    def complete(self, prompt: str) -> str:
        return complete(client=self, model=self.model, prompt=prompt)

    def complete_many(
        self, prompts: list[str], n_workers: int = os.cpu_count()
    ) -> list[str]:
        return complete_many(
            client=self, model=self.model, prompts=prompts, n_workers=n_workers
        )

    def complete_msgs(self, msgs: list[Message]) -> str:
        messages: list[dict] = extract_msgs(msgs)
        return complete_msgs(client=self, model=self.model, messages=messages)


def complete_many(
    client: OpenAIModel, model: str, prompts: list[str], n_workers: int = os.cpu_count()
) -> list[str]:
    print("Calling OpenAI API")
    with Pool(processes=n_workers) as pool:
        args = [(client, model, p) for p in prompts]
        return pool.starmap(complete, args)


def complete(client: OpenAIModel, model: str, prompt: str) -> str:
    messages = [{"role": "user", "content": prompt}]
    return complete_msgs(client=client, model=model, messages=messages)


def complete_msgs(client: OpenAIModel, model: str, messages: list[dict]) -> str:
    url = f"{client.base_url}/chat/completions"
    payload = {"model": model, "temperature": 0.0, "messages": messages}

    response = requests.post(url, headers=client.headers(), json=payload)
    response.raise_for_status()
    result = response.json()

    assert len(result["choices"]) == 1
    return result["choices"][0]["message"]["content"]


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
