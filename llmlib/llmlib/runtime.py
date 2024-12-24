from .internvl import InternVL
from .base_llm import LLM
from .gemini.media_description import GeminiAPI
from .gemma import PaliGemma2
from .minicpm import MiniCPM
from .llama3 import LLama3Vision8B
from .model_registry import ModelEntry, ModelRegistry
from .openai.openai_completion import OpenAIModel
from .phi3.phi3 import Phi3Vision


def filled_model_registry() -> ModelRegistry:
    return ModelRegistry(
        models=[
            ModelEntry.from_cls_with_id(Phi3Vision),
            ModelEntry.from_cls_with_id(MiniCPM),
            ModelEntry.from_cls_with_id(LLama3Vision8B),
            ModelEntry.from_cls_with_id(PaliGemma2),
            ModelEntry.from_cls_with_id(InternVL),
            *model_entries_from_mult_ids(OpenAIModel),
            *model_entries_from_mult_ids(GeminiAPI),
        ]
    )


def model_entries_from_mult_ids(cls: type[LLM]) -> list[ModelEntry]:
    assert hasattr(cls, "model_ids")
    entries = [
        ModelEntry(
            model_id=id_,
            clazz=cls,
            ctor=lambda: cls(model_id=id_),
            warnings=cls.get_warnings(),
        )
        for id_ in cls.model_ids
    ]
    return entries
