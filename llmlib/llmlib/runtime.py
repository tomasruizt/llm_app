from .replicate_api import Apollo7B
from .internvl import InternVL
from .gemini.media_description import GeminiAPI
from .gemma import PaliGemma2
from .minicpm import MiniCPM
from .llama3 import LLama3Vision8B
from .model_registry import ModelEntry, ModelRegistry, model_entries_from_mult_ids
from .openai.openai_completion import OpenAIModel
from .phi3.phi3 import Phi3Vision


def filled_model_registry() -> ModelRegistry:
    return ModelRegistry(
        models=[
            *model_entries_from_mult_ids(MiniCPM),
            ModelEntry.from_cls_with_id(InternVL),
            ModelEntry.from_cls_with_id(Apollo7B),
            ModelEntry.from_cls_with_id(Phi3Vision),
            ModelEntry.from_cls_with_id(LLama3Vision8B),
            ModelEntry.from_cls_with_id(PaliGemma2),
            *model_entries_from_mult_ids(OpenAIModel),
            *model_entries_from_mult_ids(GeminiAPI),
        ]
    )
