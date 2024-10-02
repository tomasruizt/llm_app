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
            *[
                ModelEntry(
                    model_id=id_, clazz=OpenAIModel, ctor=lambda: OpenAIModel(model=id_)
                )
                for id_ in OpenAIModel.model_ids
            ],
        ]
    )
