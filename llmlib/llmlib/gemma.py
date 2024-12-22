from llmlib.base_llm import LLM, Message
from transformers import PaliGemmaProcessor, PaliGemmaForConditionalGeneration
import torch


class PaliGemma2(LLM):
    # model_id = "google/paligemma2-3b-ft-docci-448"  DOCCI is more verbose but does not speak about implicit meanings in the image.
    model_id = "google/paligemma2-3b-pt-896"
    requires_gpu_exclusively = True

    def __init__(self):
        self.model = PaliGemmaForConditionalGeneration.from_pretrained(
            self.model_id, torch_dtype=torch.bfloat16, device_map="auto"
        ).eval()
        self.processor = PaliGemmaProcessor.from_pretrained(self.model_id)

    def complete_msgs2(self, msgs: list[Message]) -> str:
        if len(msgs) != 1:
            raise ValueError("Currently, Gemma2 only supports one message at a time")

        prompt = msgs[0].msg
        image = msgs[0].img
        if image is None:
            raise ValueError("Gemma2 requires an image")

        inputs = (
            self.processor(text=prompt, images=image, return_tensors="pt")
            .to(torch.bfloat16)
            .to(self.model.device)
        )
        input_len = inputs["input_ids"].shape[-1]

        with torch.inference_mode():
            generation = self.model.generate(
                **inputs, max_new_tokens=100, do_sample=False
            )
            generation = generation[0][input_len:]
            decoded: str = self.processor.decode(generation, skip_special_tokens=True)
            return decoded

    @classmethod
    def get_warnings(cls) -> list[str]:
        return [
            "This model only accepts one message by the user at a time.",
            "This model REQUIRES an image.",
            "PaliGemma2 is only pretrained and might not follow instructions well.",
        ]
