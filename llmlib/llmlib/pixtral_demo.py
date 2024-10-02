from vllm import LLM
from vllm.sampling_params import SamplingParams

if __name__ == "__main__":
    model_name = "mistralai/Pixtral-12B-2409"

    sampling_params = SamplingParams(max_tokens=8192)

    llm = LLM(model=model_name, gpu_memory_utilization=0.1, tokenizer_mode="mistral")

    prompt = "Describe this image in one sentence."
    image_url = "https://picsum.photos/id/237/200/300"

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": image_url}},
            ],
        },
    ]

    outputs = llm.chat(messages, sampling_params=sampling_params)

    print(outputs[0].outputs[0].text)
