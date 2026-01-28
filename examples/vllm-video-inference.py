"""
1. Install dependencies (tested on Python 3.11). From this directory (examples/), run:
  pip install uv
  uv pip install -e "../llmlib[all]"

2. Then run the script:
  python vllm-video-inference.py
"""

import json
from llmlib.base_llm import LlmReq, Message
from llmlib.vllm_model import ModelvLLM
from llmlib.vllmserver import spinup_vllm_server

model_id = "Qwen/Qwen2.5-VL-3B-Instruct"

# Download example video from https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct
# wget https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2-VL/space_woaudio.mp4
file = "space_woaudio.mp4"

prompts = [
    "What is shown in the video?",
    "How many people are shown in the video?",
    "Where is the video taken?",
    "What is the main subject of the video?",
    "What should the video be called?",
    "What time of day is it in the video?",
    "What emotions are expressed in the video?",
    "What objects are visible in the video?",
    "What is happening in the background of the video?",
    "What season does the video take place in?",
    "Are there any vehicles in the video?",
    "What colors are most common in the video?",
    "Is the video filmed indoors or outdoors?",
    "What kind of activity is taking place in the video?",
]

client = ModelvLLM(
    model_id=model_id,
    max_new_tokens=200,
    temperature=0.7,
    remote_call_concurrency=4,
    port=8000,
)

# The vLLM server can be started from the command line,
# or it can be started by Python, as done below.
start_vllm_server = True
vllm_command = [
    "vllm",
    "serve",
    model_id,
    "--dtype=bfloat16",
    "--port=8000",
    "--allowed-local-media-path=/",  # allow access everywhere
    "--host=127.0.0.1",  # to prevent access from other machines
    '--limit-mm-per-prompt={"image": 1, "video": 1}',
    # The values below are for an NVIDIA RTX 3090 (24GB VRAM)
    # Read more about this settings here: https://github.com/QwenLM/Qwen2.5-VL?tab=readme-ov-file#image-resolution-for-performance-boost
    # and here: https://github.com/QwenLM/Qwen2.5-VL?tab=readme-ov-file#image-resolution-for-performance-boost
    '--mm-processor-kwargs={"max_pixels": 65536, "fps": 1}',
    "--max_model_len=20000",
    "--gpu-memory-utilization=0.7",
    "--max-num-seqs=8",
]

batch = [
    LlmReq(
        convo=[Message(role="user", msg=prompt, video=file)],
        metadata={"prompt": prompt},
    )
    for prompt in prompts
]

with spinup_vllm_server(
    no_op=not start_vllm_server, vllm_command=vllm_command
) as server:
    for answer in client.complete_batchof_reqs(batch):
        with open("answers.jsonl", "at") as f:
            row = json.dumps(answer)
            f.write(row + "\n")
        print("Dumped response to file answers.jsonl")
