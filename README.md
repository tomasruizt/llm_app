# LLM Multimodal Vibe-Check
We use this streamlit app to chat with different multimodal open-source and propietary LLMs. The idea is to quickly assess qualitatively (vibe-check) whether the model understands the nuance of harmful language.

https://github.com/user-attachments/assets/2fb49053-651c-4cc9-b102-92a392a3c473

## Run Streamlit App
In the `docker-compose.yml` file, you will need to change the volume to point to your own huggingface model cache. To run the app, use the following command:
```bash
docker compose up videoapp
```

### Run Only Inference Server
```bash
docker compose up rest_api
```

## Structure
* Each multimodal LLM has a different way of consuming image(s). This codebase unifies the different interfaces e.g. of Phi-3, MinCPM, OpenAI GPT-4o, etc. This is done with a single base class `LLM` (interface) which is then implemented by each concrete model. You can find these implementation in the directory `llmlib/llmlib/`.
* The open-source implementation are based on the `transformers` library. I have experimented with `vLLM`, but it made the GPU run OOM. More fiddling is needed.
* I have extracted a REST API using `FastAPI` to decouple the frontend streamlit code from the inference server.
* The app supports small open-source models atm, because the inference server is running a single 24GB VRAM GPU. We will hopefully scale this backend up soon.

## Archive: Installation Tips
Installation for the quantized model in `llama_cpp`:
```shell
CMAKE_ARGS="-DLLAMA_CUBLAS=on -DCUDA_PATH=/usr/local/cuda-12.5 -DCUDAToolkit_ROOT=/usr/local/cuda-12.5 -DCUDAToolkit_INCLUDE_DIR=/usr/local/cuda-12/include -DCUDAToolkit_LIBRARY_DIR=/usr/local/cuda-12.5/lib64" FORCE_CMAKE=1 pip install llama-cpp-python --no-cache-dir
```