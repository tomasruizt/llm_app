.PHONY: vllm-server

vllm-server:
	vllm serve Qwen/Qwen2.5-VL-3B-Instruct \
		--task generate \
		--max-model-len 32768 \
		--max-seq-len-to-capture 32768 \
		--dtype bfloat16 \
		--allowed-local-media-path=/home/ \
		--limit-mm-per-prompt "image=50,video=2" \
		--disable-log-requests \
		--port 8000 \
		--gpu-memory-utilization 0.8
