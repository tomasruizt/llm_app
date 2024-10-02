from llmlib.base_llm import Message
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import BitsAndBytesConfig
from llmlib.base_llm import LLM
from PIL import Image

_model_id = "qresearch/llama-3-vision-alpha-hf"


class LLama3Vision8B(LLM):
    model_id = _model_id
    requires_gpu_exclusively = True

    def __init__(self):
        self.model = create_model()
        self.tokenizer = create_tokenizer()

    def complete_msgs2(self, msgs: list[Message]) -> str:
        if len(msgs) != 1:
            raise ValueError(
                f"model='{_model_id}' supports only one message by the user."
            )
        msg = msgs[0]
        if msg.role != "user":
            raise ValueError(
                f"model='{_model_id}' supports only a role=user message, not role={msg.role}."
            )

        # 2024-06-20: Model does not accept image=None, therefore we create a small white image
        if msg.img is None:
            empty_img = Image.new("RGB", (3, 3), color="white")
            image = empty_img
        else:
            image = msg.img

        response: str = self.tokenizer.decode(
            self.model.answer_question(image, msg.msg, self.tokenizer),
            skip_special_tokens=True,
        )
        return response

    @classmethod
    def get_warnings(cls) -> list[str]:
        return ["This model only accepts one message by the user at a time."]


def create_model():
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        llm_int8_skip_modules=["mm_projector", "vision_model"],
    )

    return AutoModelForCausalLM.from_pretrained(
        _model_id,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        quantization_config=bnb_cfg,
    )


def create_tokenizer():
    return AutoTokenizer.from_pretrained(
        _model_id,
        use_fast=True,
    )
