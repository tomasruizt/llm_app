from typing_extensions import Self
from dataclasses import dataclass, field
from typing import Callable
from .base_llm import LLM


@dataclass
class ModelEntry:
    model_id: str
    clazz: type[LLM]
    ctor: Callable[[], LLM]
    warnings: list[str]
    infos: list[str]

    @classmethod
    def from_cls_with_id(cls, T: type[LLM]) -> Self:
        return cls(
            model_id=T.model_id,
            clazz=T,
            ctor=T,
            warnings=T.get_warnings(),
            infos=T.get_info(),
        )


@dataclass
class ModelRegistry:
    models: list[ModelEntry] = field(default_factory=list)

    def get_entry(self, model_id: str) -> ModelEntry:
        id2entry = {entry.model_id: entry for entry in self.models}
        return id2entry[model_id]

    def all_model_ids(self) -> list[str]:
        return [entry.model_id for entry in self.models]
