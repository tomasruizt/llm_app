from dataclasses import dataclass
from pathlib import Path 
import time
from llmlib.base_llm import LLM, validate_only_first_message_has_files
import torch
from llmlib.huggingface_inference import Message, is_img, is_video, video_to_imgs
from transformers import AutoProcessor, Llama4ForConditionalGeneration


@dataclass
class Llama_4(LLM):
    model_id: str
    max_n_frames_per_video: int = 100
    max_new_tokens: int = 500

    model_ids = [
        "unsloth/Llama-4-Scout-17B-16E-Instruct",
    ]

    def __post_init__(self):
        self.model = Llama4ForConditionalGeneration.from_pretrained(
            self.model_id,
            device_map="auto",
            attn_implementation="flex_attention",
            torch_dtype=torch.bfloat16,
        ).eval()
        self.processor = AutoProcessor.from_pretrained(self.model_id)

    def complete_msgs(
        self, msgs: list[Message], output_dict: bool = False
    ) -> str | dict: 
        validate_only_first_message_has_files(msgs)

        messages: list[dict] = [
            convert_mgs_to_llama_4_format(msg, self.max_n_frames_per_video)
            for msg in msgs
        ]

        inputs = self.processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        ).to(self.model.device)
        start = time.time()
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
        )
        runtime = time.time() - start
        response = self.processor.batch_decode(outputs[:, inputs["input_ids"].shape[-1]:])[0]
        if output_dict:
            n_frames = len([c for c in messages[0]["content"] if c["type"] == "image"])
            return {
                "response": response,
                "n_frames": n_frames,
                "model_runtime": runtime,
            }
        return response

def convert_mgs_to_llama_4_format(msg: Message, max_n_frames_per_video: int) -> dict:
    dict_msg = {"role": msg.role, "content": []}
    if msg.img is not None:
        image = msg.img
        if isinstance(image, Path):
            image = str(image)
        dict_msg["content"].append({"type": "image", "image": image})
    if msg.video is not None:
        imgs: list = video_to_imgs(msg.video, max_n_frames_per_video)
        for img in imgs:
            dict_msg["content"].append({"type": "image", "image": img})
    if msg.files is not None:
        for filepath in msg.files:
            if is_img(filepath):
                dict_msg["content"].append({"type": "image", "image": str(filepath)})
            elif is_video(filepath):
                imgs: list = video_to_imgs(filepath, max_n_frames_per_video)
                for img in imgs: 
                    dict_msg["content"].append({"type": "image", "image": img})
    if msg.msg:
        dict_msg["content"].append({"type": "text", "text": msg.msg})

    return dict_msg