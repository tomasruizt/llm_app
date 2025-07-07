.PHONY: vllm-server
.PHONY: whisper-vllm

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

whisper-vllm:
	VLLM_USE_V1=0 python -m vllm.entrypoints.openai.api_server \
		--model openai/whisper-large-v3-turbo \
		--task transcription \
		--allowed-local-media-path=/ \
		--limit-mm-per-prompt "audio=1" \
		--disable-log-requests \
		--host 127.0.0.1 \
		--port 9000 \