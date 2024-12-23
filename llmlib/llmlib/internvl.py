from llmlib.base_llm import LLM, Message
from lmdeploy import pipeline


class InternVL(LLM):
    model_id = "OpenGVLab/InternVL2_5-8B-AWQ"

    def __init__(self):
        self.pipe = pipeline(self.model_id)

    def complete_msgs2(self, msgs: list[Message]) -> str:
        if len(msgs) != 1:
            raise ValueError("InternVL only supports one message")
        imgs = []
        if msgs[0].has_image():
            imgs.append(msgs[0].img)
        return self.pipe((msgs[0].msg, imgs)).text
