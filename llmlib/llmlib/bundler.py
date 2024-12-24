from dataclasses import dataclass, field
import logging

from .bundler_request import BundlerRequest
from .model_registry import ModelEntry, ModelRegistry
from .base_llm import LLM
import torch
import gc


logger = logging.getLogger(__name__)


@dataclass
class Bundler:
    """Makes sure that only 1 model occupies the GPU at a time."""

    registry: ModelRegistry = field(default_factory=ModelRegistry)
    model_on_gpu: LLM | None = None
    id2_nongpu_model: dict[str, LLM] = field(default_factory=dict)

    def id_of_model_on_gpu(self) -> str | None:
        return None if self.model_on_gpu is None else self.model_on_gpu.model_id

    def get_response(self, req: BundlerRequest) -> str:
        e: ModelEntry = self.registry.get_entry(model_id=req.model_id)
        model: LLM = self._get_model_instance(e=e)
        last_msg = req.msgs[-1]
        if last_msg.has_video():
            if len(req.msgs) > 1:
                raise ValueError("Video only supported for single message requests")
            return model.video_prompt(last_msg.video, last_msg.msg)
        return model.complete_msgs2(req.msgs)

    def _get_model_instance(self, e: ModelEntry) -> LLM:
        if e.clazz.requires_gpu_exclusively:
            self.set_model_on_gpu(model_id=e.model_id)
            model: LLM = self.model_on_gpu
        else:
            if e.model_id not in self.id2_nongpu_model:
                self.id2_nongpu_model[e.model_id] = e.ctor()
            model: LLM = self.id2_nongpu_model[e.model_id]
        return model

    def set_model_on_gpu(self, model_id: str) -> None:
        if (
            self.id_of_model_on_gpu() is not None
            and self.id_of_model_on_gpu() == model_id
        ):
            return
        assert model_id in self.registry.all_model_ids()

        e: ModelEntry = self.registry.get_entry(model_id)
        if not e.clazz.requires_gpu_exclusively:
            logger.info(
                "Model does not require GPU exclusively. Ignoring set_model_on_gpu() call."
            )
            return

        self.clear_model_on_gpu()
        self.model_on_gpu = e.ctor()

    def clear_model_on_gpu(self):
        self.model_on_gpu = None
        gc.collect()
        torch.cuda.empty_cache()
