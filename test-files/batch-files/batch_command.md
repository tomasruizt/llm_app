# Batch Inference

To generate the batch output, the `batch_input.jsonl` must exist. Then run this command from this directory, so that the `../test-files/` path is correct:

```bash
python -m vllm.entrypoints.openai.run_batch -i batch_input.jsonl -o results.jsonl --model Qwen/Qwen2.5-VL-3B-Instruct --allowed-local-media-path=/home/tomasruiz/code/llm_app/test-files --enforce-eager
```
