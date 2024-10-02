from dataclasses import dataclass
from llmlib.bundler import Bundler
from llmlib.bundler_request import BundlerRequest
from llmlib.base_llm import LLM, Message
import pytest
from llmlib.model_registry import ModelEntry, ModelRegistry


def test_model_id_on_gpu():
    b = Bundler(filled_model_registry())
    assert b.id_of_model_on_gpu() is None
    b.set_model_on_gpu(GpuLLM.model_id)
    assert b.id_of_model_on_gpu() == GpuLLM.model_id


def test_get_response():
    b = Bundler(filled_model_registry())
    msgs = [Message(role="user", msg="hello")]
    request = BundlerRequest(model_id=GpuLLM.model_id, msgs=msgs)
    expected_response = GpuLLM().complete_msgs2(msgs)
    actual_response: str = b.get_response(request)
    assert actual_response == expected_response
    assert b.id_of_model_on_gpu() == GpuLLM.model_id


def test_bundler_multiple_responses():
    b = Bundler(filled_model_registry())
    models = [GpuLLM(), GpuLLM2(), NonGpuLLM()]
    msgs = [Message(role="user", msg="hello")]

    expected_responses = [m.complete_msgs2(msgs) for m in models]
    assert expected_responses[0] != expected_responses[1]

    actual_responses = [
        b.get_response(BundlerRequest(model_id=m.model_id, msgs=msgs)) for m in models
    ]
    assert actual_responses == expected_responses

    last_gpu_model = [m for m in models if m.requires_gpu_exclusively][-1]
    assert b.id_of_model_on_gpu() == last_gpu_model.model_id


def test_set_model_on_gpu():
    b = Bundler(filled_model_registry())
    b.set_model_on_gpu(GpuLLM.model_id)
    assert b.id_of_model_on_gpu() == GpuLLM.model_id

    with pytest.raises(AssertionError):
        b.set_model_on_gpu("invalid")
    assert b.id_of_model_on_gpu() == GpuLLM.model_id

    b.set_model_on_gpu(NonGpuLLM.model_id)
    gpu_model_is_still_loaded: bool = b.id_of_model_on_gpu() == GpuLLM.model_id
    assert gpu_model_is_still_loaded


def filled_model_registry() -> ModelRegistry:
    model_entries = [
        ModelEntry.from_cls_with_id(GpuLLM),
        ModelEntry.from_cls_with_id(GpuLLM2),
        ModelEntry.from_cls_with_id(NonGpuLLM),
    ]
    return ModelRegistry(model_entries)


@dataclass
class GpuLLM(LLM):
    model_id = "gpu-llm-model"
    requires_gpu_exclusively = True

    def complete_msgs2(self, msgs: list[Message]) -> str:
        return "gpu msg"


@dataclass
class GpuLLM2(LLM):
    model_id = "gpu-llm-model-2"
    requires_gpu_exclusively = True

    def complete_msgs2(self, msgs: list[Message]) -> str:
        return "gpu msg 2"


@dataclass
class NonGpuLLM(LLM):
    model_id = "non-gpu-llm-model"
    requires_gpu_exclusively = False

    def complete_msgs2(self, msgs: list[Message]) -> str:
        return "non-gpu message"
