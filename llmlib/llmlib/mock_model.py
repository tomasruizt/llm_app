from .base_llm import LLM, LlmReq


from typing import Iterable


class MockModel(LLM):
    model_id = "mock-model"

    def complete_batchof_reqs(self, batch: Iterable[LlmReq]) -> Iterable[dict]:
        for req in batch:
            yield {
                "success": True,
                "response": "mock response",
                "reasoning": "mock reasoning",
                **req.gen_kwargs,
                **req.metadata,
            }
