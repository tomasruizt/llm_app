from dataclasses import dataclass
from llmlib.base_llm import LLM
from llmlib.model_registry import model_entries_from_mult_ids


@dataclass
class LLMForTest(LLM):
    model_id: str
    model_ids = ["id1", "id2"]


def test_model_entries_from_mult_ids():
    e1, e2 = model_entries_from_mult_ids(LLMForTest)
    assert e1.model_id == "id1"
    assert e2.model_id == "id2"

    llm1 = e1.ctor()
    llm2 = e2.ctor()
    assert llm1.model_id == "id1"
    assert llm2.model_id == "id2"
