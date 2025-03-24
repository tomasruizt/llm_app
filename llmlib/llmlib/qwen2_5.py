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
        image = msg.image
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




        # batch_inference differs from gemma where you zip input and output ids 

        # define convert msgs function according to the prompt of qwen

        # here use below will-be-defined function to format according to qwen

        # model specific init, processor, pass input and get output 
        # using VideoOutput dataclass as bridge inside run_benchmark.py
        # todo: in run_benchmark.py 
        # implement qwen loader without tokenizer 
        # implement qwen dataclass and bridge them with VideoOutput pass in process_video function to get benchmark results
        # try out with cli arguments 

        # question to ask: what happened to our framing functions? 
        # you might need to go back to them since 72b!!



#gemma
# messages = [
#     {
#         "role": "system",
#         "content": [{"type": "text", "text": "You are a helpful assistant."}]
#     },
#     {
#         "role": "user",
#         "content": [
#             {"type": "image", "image": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bee.jpg"},
#             {"type": "text", "text": "Describe this image in detail."}
#         ]
#     }
# ]

#qwen, also try out video inference one
# Messages containing multiple images and a text query
# messages = [
#     {
#         "role": "user",
#         "content": [
#             {"type": "image", "image": "file:///path/to/image1.jpg"},
#             {"type": "image", "image": "file:///path/to/image2.jpg"},
#             {"type": "text", "text": "Identify the similarities between these images."},
#         ],
#     }
# ]


# dependencies:
# The code of Qwen2.5-VL has been in the latest Hugging face transformers and we advise you to build from source with command:

# pip install git+https://github.com/huggingface/transformers accelerate

# or you might encounter the following error:

# KeyError: 'qwen2_5_vl'

# We offer a toolkit to help you handle various types of visual input more conveniently, as if you were using an API. This includes base64, URLs, and interleaved images and videos. You can install it using the following command:

# # It's highly recommanded to use `[decord]` feature for faster video loading.
# pip install qwen-vl-utils[decord]==0.0.8

# If you are not using Linux, you might not be able to install decord from PyPI. In that case, you can use pip install qwen-vl-utils which will fall back to using torchvision for video processing. However, you can still install decord from source to get decord used when loading video.


# pip install qwen_vl_utils
# put the model into llmlib just import in run_benchmark.py
# model specifics no tokenizer, instead use post_init like in gemma3local to return with processor initialized
# create dataclass inside run_benchmark.py like minicpm and gemma3local set llmlib_model_id and import necessary functionalities
# load_model accordingly

# what else you need 
# adapt the following fucntions from gemma3local to qwen
# 
# func
# __post_init__
# func
# complete_msgs
# func
# convert_msg_to_gemma3_for

