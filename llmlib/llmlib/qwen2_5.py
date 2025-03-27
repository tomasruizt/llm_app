from dataclasses import dataclass
from pathlib import Path 
import time
from llmlib.base_llm import LLM, validate_only_first_message_has_files
import torch
from llmlib.huggingface_inference import Message, is_img, is_video, video_to_imgs
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info

@dataclass
class Qwen2_5(LLM):
    model_id: str
    max_n_frames_per_video: int = 100
    max_new_tokens: int = 500

    model_ids = [
        "Qwen/Qwen2.5-VL-3B-Instruct",
        "Qwen/Qwen2.5-VL-7B-Instruct",
        "Qwen/Qwen2.5-VL-72B-Instruct",
    ]

    def __post_init__(self):
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_id,
            device_map="auto",
            attn_implementation="flash_attention_2",
            torch_dtype=torch.bfloat16,
        ).eval()
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        
    def complete_msgs(
        self, msgs: list[Message], output_dict: bool = False
    ) -> str | dict: 
        validate_only_first_message_has_files(msgs)

        messages: list[dict] = [
            convert_mgs_to_qwen_2_5_format(msg, self.max_n_frames_per_video)
            for msg in msgs
        ]

        # prep for inference
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.model.device)

        start = time.time()
        generated_ids = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)
        runtime = time.time() - start
        generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        response: str = self.processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        if output_dict:
            n_frames = len([c for c in messages[0]["content"] if c["type"] == "image"])
            return {
                "response": response,
                "n_frames": n_frames,
                "model_runtime": runtime,
            }
        return response

def convert_mgs_to_qwen_2_5_format(msg: Message, max_n_frames_per_video: int) -> dict:
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