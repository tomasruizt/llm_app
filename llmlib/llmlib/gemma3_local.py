from dataclasses import dataclass
from pathlib import Path
from llmlib.base_llm import LLM, validate_only_first_message_has_files
import torch
from llmlib.huggingface_inference import Message, is_img, is_video, video_to_imgs
from transformers import AutoProcessor, Gemma3ForConditionalGeneration


@dataclass
class Gemma3Local(LLM):
    model_id: str
    max_n_frames_per_video: int = 100
    max_new_tokens: int = 500

    model_ids = [
        "google/gemma-3-4b-it",
        "google/gemma-3-27b-it",
    ]

    def __post_init__(self):
        self.model = Gemma3ForConditionalGeneration.from_pretrained(
            self.model_id,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        ).eval()
        self.processor = AutoProcessor.from_pretrained(self.model_id)

    def complete_msgs(self, msgs: list[Message]) -> str:
        """Complete a conversation with the model."""
        validate_only_first_message_has_files(msgs)

        messages: list[dict] = [
            convert_msg_to_gemma3_format(msg, self.max_n_frames_per_video)
            for msg in msgs
        ]

        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)

        with torch.inference_mode():
            outputs = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)

        input_len = len(inputs["input_ids"][0])
        response: str = self.processor.decode(
            outputs[0][input_len:], skip_special_tokens=True
        )
        return response


def convert_msg_to_gemma3_format(msg: Message, max_n_frames_per_video: int) -> dict:
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
