"""Audio-input VLM sanity check served via vLLM.

Exercises the audio path end-to-end: standalone .flac/.mp3 files must reach
the model's audio encoder. The fixture's `vllm_audio_test_command` documents
how to spin up the server (mistral tokenizer + audio mm-limit); the actual
server is started externally and the test points at it.

To run locally with Voxtral-Mini-3B:
    LD_LIBRARY_PATH=$(python -c 'import nvidia.cu13.lib as _; \
        import os; print(os.path.dirname(_.__file__))'):$LD_LIBRARY_PATH \
    vllm serve RedHatAI/Voxtral-Mini-3B-2507-FP8-dynamic \
      --port=8001 --max-model-len=16384 --dtype=bfloat16 \
      --allowed-local-media-path=/home/ \
      --limit-mm-per-prompt='{"audio":4}' \
      --tokenizer-mode mistral --config-format mistral --load-format mistral \
      --enforce-eager
"""

import pytest
from llmlib.vllm_model import ModelvLLM
from llmlib.vllmserver import VLLMServer, spinup_vllm_server

from .helpers import assert_model_can_hear_audio, is_ci


model_id = "RedHatAI/Voxtral-Mini-3B-2507-FP8-dynamic"
port = 8001


def vllm_audio_test_command(model_id: str) -> list[str]:
    return [
        "vllm",
        "serve",
        model_id,
        "--max-model-len=16384",
        "--dtype=bfloat16",
        "--allowed-local-media-path=/home/",
        '--limit-mm-per-prompt={"audio":4}',
        f"--port={port}",
        "--gpu-memory-utilization=0.85",
        "--max-num-seqs=4",
        "--enforce-eager",
        "--tokenizer-mode=mistral",
        "--config-format=mistral",
        "--load-format=mistral",
    ]


@pytest.fixture(scope="session")
def vllm_audio_server():
    cmd = vllm_audio_test_command(model_id)
    with spinup_vllm_server(no_op=True, vllm_command=cmd) as server:
        yield server


@pytest.fixture(scope="session")
def vllm_audio_model(vllm_audio_server: VLLMServer) -> ModelvLLM:
    return ModelvLLM(
        model_id=model_id,
        port=port,
        timeout_secs=120,
        max_new_tokens=200,
        temperature=0.0,
    )


@pytest.mark.skipif(condition=is_ci(), reason="Requires local GPU + vLLM")
def test_vllm_audio_model_can_hear_audio(vllm_audio_model: ModelvLLM):
    assert_model_can_hear_audio(vllm_audio_model)
