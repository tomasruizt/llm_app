.PHONY: vllm-server

vllm-server:
	python -m vllm.entrypoints.openai.api_server \
		--model Qwen/Qwen2.5-VL-3B-Instruct \
		--task generate \
		--max-model-len 32768 \
		--max-seq-len-to-capture 32768 \
		--dtype bfloat16 \
		--allowed-local-media-path=/ \
		--limit-mm-per-prompt "image=50,video=2" \
		--disable-log-requests \
		--port 8000 \
		--host 127.0.0.1 \
		--gpu-memory-utilization 0.8 \
		--enforce-eager
