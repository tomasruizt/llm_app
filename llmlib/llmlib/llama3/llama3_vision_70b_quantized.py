from pathlib import Path
from llama_cpp import Llama
from llama_cpp.llama_chat_format import Llama3VisionAlpha
from llmlib.base_llm import Message
from llmlib.openai.openai_completion import extract_msgs
from llmlib.base_llm import LLM
from PIL import Image


class LLama3Vision70BQuantized(LLM):
    model_id = "Meta-Llama-3-70B-Instruct-IQ1_M.gguf"
    requires_gpu_exclusively = True

    def __init__(self) -> None:
        clip_path = Path(__file__).parent / "models/mmproj-model-f16.gguf"
        assert clip_path.exists()
        chat_handler = Llama3VisionAlpha(clip_model_path=str(clip_path), verbose=True)
        self.llm = Llama.from_pretrained(
            repo_id="lmstudio-community/Meta-Llama-3-70B-Instruct-GGUF",
            filename=self.model_id,
            verbose=True,
            n_gpu_layers=-1,
            flash_attn=True,  # not sure if this changed anything
            n_ctx=2048,  # arbitrary, can be set higher if needed
            # without chat_handler & chat_format, the model tried to parse the img as text.
            chat_handler=chat_handler,
            chat_format="llama-3-vision-alpha",
        )

    def complete_msgs2(self, msgs: list[Message]) -> str:
        messages: list[dict] = extract_msgs(msgs)
        return self.complete_msgs(messages=messages)

    def complete_msgs(
        self, messages: list[dict], images: list[Image.Image] = []
    ) -> str:
        cs = self.llm.create_chat_completion(messages=messages)
        return cs["choices"][0]["message"]["content"]

    @classmethod
    def get_warnings(cls) -> list[str]:
        return [
            "This model is quantized to 1bit, performance could be strongly degraded",
            "This model does not yet integrate images gracefully.",
        ]


if __name__ == "__main__":
    # Run like this:
    #     cd llm_app
    #     python -m llama3.llama3_quantized_example
    llm = LLama3Vision70BQuantized()
    msgs = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France?"},
    ]
    print("response:", llm.complete_msgs(messages=msgs))
