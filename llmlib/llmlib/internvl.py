from llmlib.base_llm import LLM, Message
from lmdeploy import pipeline, VisionConfig


class InternVL(LLM):
    model_id = "OpenGVLab/InternVL2_5-8B-AWQ"

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
