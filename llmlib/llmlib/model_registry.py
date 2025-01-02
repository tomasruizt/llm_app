from typing_extensions import Self
from dataclasses import dataclass, field
from typing import Any
from .base_llm import LLM


@dataclass
class ModelEntry:
    model_id: str
    clazz: type[LLM]
    ctor_kwargs: dict[str, Any]
    warnings: list[str]
    infos: list[str]

    @classmethod
    def from_cls_with_id(cls, T: type[LLM]) -> Self:
        return cls(
            model_id=T.model_id,
            clazz=T,
            ctor_kwargs={},
            warnings=T.get_warnings(),
            infos=T.get_info(),
        )

    def ctor(self) -> LLM:
        return self.clazz(**self.ctor_kwargs)


@dataclass
class ModelRegistry:
    models: list[ModelEntry] = field(default_factory=list)

    def get_entry(self, model_id: str) -> ModelEntry:
        id2entry = {entry.model_id: entry for entry in self.models}
        return id2entry[model_id]

    def all_model_ids(self) -> list[str]:
        return [entry.model_id for entry in self.models]


def model_entries_from_mult_ids(cls: type[LLM]) -> list[ModelEntry]:
    assert hasattr(cls, "model_ids")
    entries = [
        ModelEntry(
            model_id=id_,
            clazz=cls,
            ctor_kwargs={"model_id": id_},
            warnings=cls.get_warnings(),
            infos=cls.get_info(),
        )
        for id_ in cls.model_ids
    ]
    return entries
