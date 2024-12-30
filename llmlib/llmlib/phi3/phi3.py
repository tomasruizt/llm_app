from dataclasses import dataclass
from typing import Any
from llmlib.base_llm import Message
from torch import Tensor
import torch
from transformers import AutoModelForCausalLM, AutoProcessor
from PIL import Image
from llmlib.base_llm import LLM
from transformers.image_processing_utils import BatchFeature

model_id = "microsoft/Phi-3.5-vision-instruct"


@dataclass
class GenConf:
    max_new_tokens: int = 500
    temperature: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        do_sample: bool = self.temperature != 0.0
        return {
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature if do_sample else None,
            "do_sample": do_sample,
        }


class Phi3Vision(LLM):
    model_id = model_id
    requires_gpu_exclusively = True

    def __init__(self, gen_conf: GenConf | None = None):
        self.model = create_model()
        self.processor = create_processor()
        self.gen_conf = GenConf() if gen_conf is None else gen_conf

    def complete(self, prompt: str) -> str:
        msg = Message(role="user", msg=prompt)
        return completion(llm=self, batch=[[msg]])[0]

    def complete_msgs(self, msgs: list[Message]) -> str:
        return completion(llm=self, batch=[msgs])[0]

    def complete_batch(self, batch: list[list[Message]]) -> list[str]:
        return completion(llm=self, batch=batch)


def extract_imgs_and_dicts(msgs: list[Message]) -> tuple[list[Image.Image], list[dict]]:
    """
    Phi3 expects in the prompts placehodlers for images in the form <|image_X|>, where X is the image number.
    It also requires the images as a separate array of PIL images.
    This function extracts the images from the messages and creates the placeholders.
    It makes sure to avoid duplication in the images and placeholders.
    """
    img_names = list(dict.fromkeys(m.img_name for m in msgs if m.img_name is not None))
    placeholders = {
        img_name: f"<|image_{i}|>" for i, img_name in enumerate(img_names, 1)
    }
    imgs = {}
    for msg in msgs:
        if msg.img is not None and msg.img_name not in imgs:
            imgs[msg.img_name] = msg.img
    images = list(imgs.values())

    messages: list[dict] = []  # entries are {"role": str, "content": str}
    for m in msgs:
        if m.img is not None and m.img_name is not None:
            img_placeholder = placeholders[m.img_name]
            content = f"{img_placeholder}\n{m.msg}"
        else:
            content = m.msg
        messages.append({"role": m.role, "content": content})
    return images, messages


def create_model(model_id: str = model_id):
    return AutoModelForCausalLM.from_pretrained(
        model_id, device_map="cuda", trust_remote_code=True, torch_dtype="auto"
    )


def create_processor(model_id: str = model_id):
    return AutoProcessor.from_pretrained(model_id, trust_remote_code=True)


def convert_to_messages(prompts: list[str]) -> list[list[dict]]:
    return [[{"role": "user", "content": prompt}] for prompt in prompts]


def completion(llm: Phi3Vision, batch: list[list[Message]]) -> list[str]:
    reject_invalid_batches(batch)
    listof_inputs: list[BatchFeature] = []
    for messages in batch:
        images, messages_dicts = extract_imgs_and_dicts(messages)
        prompt: str = llm.processor.tokenizer.apply_chat_template(
            messages_dicts, tokenize=False, add_generation_prompt=True
        )
        imgs = None if len(images) == 0 else images
        inputs = llm.processor(prompt, imgs, return_tensors="pt").to("cuda")
        listof_inputs.append(inputs)

    pad_token_id = llm.processor.tokenizer.pad_token_id
    inputs = stack_and_pad_inputs(listof_inputs, pad_token_id=pad_token_id)

    generate_ids: Tensor = llm.model.generate(
        **inputs,
        eos_token_id=llm.processor.tokenizer.eos_token_id,
        **llm.gen_conf.to_dict(),
    )
    # the prompt is included in the output, so we need to drop it.
    generate_ids = generate_ids[:, inputs["input_ids"].shape[1] :]

    responses: list[str] = llm.processor.batch_decode(
        generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return responses


def reject_invalid_batches(batch: list[list[Message]]) -> None:
    """
    Valid batches are:
    - batch of lenght 1, or
    - batches with only a single message per entry, AND
        - all messages have an image, or
        - all messages are text only.
    """
    if len(batch) <= 1:
        return
    if any(len(msgs) != 1 for msgs in batch):
        raise ValueError("Batch must contain only one message per entry.")
    any_msg_has_img = any(msg.img is not None for msgs in batch for msg in msgs)
    any_msg_is_no_img = any(msg.img is None for msgs in batch for msg in msgs)
    if any_msg_has_img and any_msg_is_no_img:
        raise ValueError("Batch must contain an image in every entry or none at all.")


def pad_left(seqs: list[torch.Tensor], pad_token_id: int) -> torch.Tensor:
    max_len = max(len(seq) for seq in seqs)
    padded = torch.full((len(seqs), max_len), pad_token_id)
    for i, seq in enumerate(seqs):
        padded[i, -len(seq) :] = seq
    return padded


def stack_and_pad_inputs(inputs: list[BatchFeature], pad_token_id: int) -> BatchFeature:
    listof_input_ids = [i.input_ids[0] for i in inputs]
    new_input_ids = pad_left(listof_input_ids, pad_token_id=pad_token_id)
    data = dict(
        input_ids=new_input_ids,
        attention_mask=(new_input_ids != pad_token_id).long(),
    )
    has_imgs: bool = "pixel_values" in inputs[0]
    if has_imgs:
        data["pixel_values"] = torch.cat([i.pixel_values for i in inputs], dim=0)
        data["image_sizes"] = torch.cat([i.image_sizes for i in inputs], dim=0)

    return BatchFeature(data).to("cuda")
