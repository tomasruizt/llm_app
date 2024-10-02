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
* Each multimodal LLM has a different way of consuming image(s). This codebase unifies the different interfaces e.g. of Phi-3, MinCPM, OpenAI GPT-4o, etc. This is done with a single base class `LLM` (interface) which is then implemented by each concrete model. 
* I have extracted a REST API using `FastAPI` to decouple the frontend streamlit code from the inference server.
* The app supports small open-source models atm, because the inference server is running a single 24GB VRAM GPU. We will hopefully scale this backend up soon.

