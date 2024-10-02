from .base_llm import Message


from dataclasses import dataclass


@dataclass
class BundlerRequest:
    model_id: str
    msgs: list[Message]
