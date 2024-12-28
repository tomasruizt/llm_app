from llmlib.base_llm import LLM, Message
from lmdeploy import pipeline, VisionConfig


class InternVL(LLM):
    model_id = "OpenGVLab/InternVL2_5-8B-AWQ"
    requires_gpu_exclusively = True

    def __init__(self):
        self.pipe = pipeline(
            self.model_id,
            vision_config=VisionConfig(thread_safe=True),
        )

    def complete_msgs2(self, msgs: list[Message]) -> str:
        session = None
        for msg in msgs:
            imgs = []
            if msg.has_image():
                imgs.append(msg.img)
            session = self.pipe.chat((msg.msg, imgs), session=session)
        return session.response.text

    @staticmethod
    def get_info() -> list[str]:
        return [
            "Model: InternVL 2.5 8B by OpenGVLab (Shanghai AI Laboratory), HuggingFace link: [here](https://huggingface.co/OpenGVLab/InternVL2_5-8B-AWQ)",
            "This model supports images and multi-turn conversations.",
            "This model is quantized to 4-bit precision using AWQ.",
        ]
