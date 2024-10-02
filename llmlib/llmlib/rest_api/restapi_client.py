import base64
import io
import logging
import os
import requests
from PIL import Image
from ..base_llm import Message
from ..bundler_request import BundlerRequest
from pydantic import BaseModel
from typing import Literal

logger = logging.getLogger(__name__)


def encode_as_png_in_base64(img: Image.Image) -> str:
    stream = io.BytesIO()
    img.save(stream, format="PNG")
    return base64.b64encode(stream.getvalue()).decode("utf-8")


class MsgDto(BaseModel):
    role: Literal["user", "assistant"]
    msg: str
    img_name: str | None = None
    img_str: str | None = None

    @classmethod
    def from_bundler_msg(cls, msg: Message) -> "MsgDto":
        return cls(
            role=msg.role,
            msg=msg.msg,
            img_name=msg.img_name,
            img_str=encode_as_png_in_base64(msg.img) if msg.img is not None else None,
        )


def to_bundler_msg(msg: MsgDto) -> Message:
    return Message(
        role=msg.role,
        msg=msg.msg,
        img_name=msg.img_name,
        img=Image.open(io.BytesIO(base64.b64decode(msg.img_str)))
        if msg.img_str
        else None,
    )


class RequestDto(BaseModel):
    model: str
    msgs: list[MsgDto]

    @classmethod
    def from_bundler_request(cls, breq: BundlerRequest) -> "RequestDto":
        return cls(
            model=breq.model_id,
            msgs=[MsgDto.from_bundler_msg(msg) for msg in breq.msgs],
        )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "model": "microsoft/Phi-3-vision-128k-instruct",
                    "msgs": [{"role": "user", "msg": "What is the capital of France?"}],
                }
            ]
        }
    }


_api_host = os.environ.get("LLMS_REST_API_HOST", "http://localhost") + ":8030"


def _headers():
    return {"X-API-Key": os.environ["LLMS_REST_API_KEY"]}


def get_completion_from_rest_api(
    breq: BundlerRequest, source=requests, **kwargs
) -> requests.Response:
    req = RequestDto.from_bundler_request(breq)
    url = _api_host + "/completion/"
    logger.info(f"Sending completion request to '{url}'.")
    return source.post(
        url=url,
        json=req.model_dump(),
        headers=_headers(),
        **kwargs,
    )


def get_models(source=requests) -> requests.Response:
    return source.get(url=_api_host + "/models/", headers=_headers())


def clear_gpu(source=requests) -> requests.Response:
    return source.post(url=_api_host + "/clear-gpu/", headers=_headers())
