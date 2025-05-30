from pathlib import Path
from typing import Any, Iterable
import pandas as pd
from dataclasses import dataclass
from llmlib.base_llm import LLM as BaseLLM, Conversation
from llmlib.huggingface_inference import is_img, is_video
import logging
from .openai.openai_completion import (
    to_synchronous_generator,
    _batch_call_openai,
)

logger = logging.getLogger(__name__)


@dataclass
class ModelvLLM(BaseLLM):
    """OpenAI-compatible vLLM server: https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html"""

    model_id: str  # e.g "google/gemma-3-4b-it"
    max_new_tokens: int = 500
    temperature: float = 0
    remote_call_concurrency: int = 8
    port: int = 8000
    timeout_secs: int = 120

    def complete_msgs(
        self, msgs: Conversation, output_dict: bool = False, **generate_kwargs
    ) -> str | dict:
        for result in self.complete_batch([msgs], **generate_kwargs):
            pass  # Avoid RuntimeError: async generator ignored GeneratorExit
        if output_dict:
            return result
        else:
            return result["response"]

    def complete_batch(
        self,
        batch: Iterable[Conversation],
        metadatas: Iterable[dict] | None = None,
        **generate_kwargs,
    ) -> Iterable[dict]:
        listof_convos = (to_vllm_oai_format(convo) for convo in batch)

        params = dict(
            model=self.model_id,
            temperature=self.temperature,
            max_tokens=self.max_new_tokens,
        )
        params = params | generate_kwargs

        server = f"http://localhost:{self.port}/v1"
        agen = _batch_call_openai(
            base_url=server,
            headers={"Content-Type": "application/json"},
            iterof_messages=listof_convos,
            gen_kwargs=params,
            remote_call_concurrency=self.remote_call_concurrency,
            timeout_secs=self.timeout_secs,
            metadatas=metadatas,
        )
        gen = to_synchronous_generator(agen)
        return gen


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
    dataset: list[Conversation], model_id: str, tgt_jsonl: Path, **gen_kwargs
):
    entries = []
    for idx, convo in enumerate(dataset):
        oai_format = to_vllm_oai_format(convo)
        entry: dict = to_batch_entry(
            oai_format,
            model_id=model_id,
            custom_id=str(idx),
            **gen_kwargs,
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
