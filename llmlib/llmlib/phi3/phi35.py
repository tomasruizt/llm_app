from PIL import Image
from llmlib.phi3.phi3 import stack_and_pad_inputs
import requests
from transformers import AutoModelForCausalLM
from transformers import AutoProcessor
from transformers.image_processing_utils import BatchFeature

model_id = "microsoft/Phi-3.5-vision-instruct"


model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="cuda",
    trust_remote_code=True,
    torch_dtype="auto",
    _attn_implementation="flash_attention_2",
)

# for best performance, use num_crops=4 for multi-frame, num_crops=16 for single-frame.
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True, num_crops=4)

links = [
    "https://image.slidesharecdn.com/azureintroduction-191206101932/75/Introduction-to-Microsoft-Azure-Cloud-1-2048.jpg",
    "https://image.slidesharecdn.com/azureintroduction-191206101932/75/Introduction-to-Microsoft-Azure-Cloud-2-2048.jpg",
    "https://image.slidesharecdn.com/azureintroduction-191206101932/75/Introduction-to-Microsoft-Azure-Cloud-3-2048.jpg",
]
images = [Image.open(requests.get(link, stream=True).raw) for link in links]
batch = [
    [{"role": "user", "content": "<|image_1|>Who is mentioned in this picture?"}],
    [{"role": "user", "content": "<|image_1|>What is the title of this image?"}],
    [{"role": "user", "content": "<|image_1|>What icons are shown in this image?"}],
]
# batch = [
#     [{"role": "user", "content": "What is the capital of France?"}],
#     [{"role": "user", "content": "How does one make a cookie that is vegetarian?"}],
# ]
# images = [None, None]

# BatchFeature(s) are the output of the processor, which is used as input to the model.
listof_inputs: list[BatchFeature] = []
for messages, image in zip(batch, images):
    prompt = processor.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    images_ = None if image is None else [image]
    inputs = processor(prompt, images_, return_tensors="pt").to("cuda:0")
    listof_inputs.append(inputs)


inputs = stack_and_pad_inputs(
    listof_inputs, pad_token_id=processor.tokenizer.pad_token_id
)

generation_args = {
    "max_new_tokens": 1000,
    "temperature": None,
    "do_sample": False,
}

generate_ids = model.generate(
    **inputs, eos_token_id=processor.tokenizer.eos_token_id, **generation_args
)

generate_ids = generate_ids[:, inputs["input_ids"].shape[1] :]
responses: list[str] = processor.batch_decode(
    generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
)

for p, r in zip(batch, responses):
    print(p)
    print(r)
    print()
