from pathlib import Path
from PIL import Image
from vllm.engine.arg_utils import EngineArgs
from transformers import AutoProcessor
from vllm import SamplingParams, LLM
from dataclasses import asdict, dataclass
from llmlib.base_llm import LLM as BaseLLM, Conversation, Message
from llmlib.huggingface_inference import is_img, video_to_imgs, is_video


@dataclass
class Gemma3vLLM(BaseLLM):
    """Inspired by https://github.com/vllm-project/vllm/blob/main/examples/offline_inference/vision_language_multi_image.py"""

    model_id: str = "google/gemma-3-4b-it"
    max_n_frames_per_video: int = 100
    max_new_tokens: int = 500

    def __post_init__(self):
        engine_args = EngineArgs(
            model=self.model_id,
            task="generate",
            max_model_len=8192,
            max_num_seqs=2,
            limit_mm_per_prompt={"image": self.max_n_frames_per_video},
            dtype="bfloat16",
        )
        engine_args = asdict(engine_args)
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.llm = LLM(**engine_args)

    def complete_batch(self, batch: list[Conversation]) -> list[str]:
        assert all(
            len(convo) == 1 for convo in batch
        ), "Each convo must have exactly one message"
        listof_inputs: list[dict] = []
        for convo in batch:
            inputs: dict = to_vllm_format(
                self.processor,
                message=convo[0],
                max_n_frames_per_video=self.max_n_frames_per_video,
            )
            listof_inputs.append(inputs)

        outputs = self.llm.generate(
            listof_inputs,
            sampling_params=SamplingParams(temperature=1.0, max_tokens=128),
        )
        for o in outputs:
            request_id = o.request_id
            n_input_tokens = len(o.prompt_token_ids)
            n_output_tokens = len(o.outputs[0].token_ids)
            print(f"{request_id=}, {n_input_tokens=}, {n_output_tokens=}")

        return [o.outputs[0].text for o in outputs]


def to_vllm_format(
    processor: AutoProcessor, message: Message, max_n_frames_per_video: int
) -> dict:
    question = message.msg
    imgs = convert_media_to_listof_imgs(message, max_n_frames_per_video)

    placeholders = [
        {"type": "image", "image": f"{idx}.jpeg"} for idx in range(len(imgs))
    ]
    messages = [
        {"role": "user", "content": [*placeholders, {"type": "text", "text": question}]}
    ]

    prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    dict_input = {
        "prompt": prompt,
        "multi_modal_data": {"image": imgs},
    }
    return dict_input


def convert_media_to_listof_imgs(
    msg: Message, max_n_frames_per_video: int
) -> list[Image.Image]:
    imgs = []
    if msg.img is not None:
        if isinstance(msg.img, (str, Path)):
            imgs.append(Image.open(msg.img))
        else:
            imgs.append(msg.img)

    if msg.video is not None:
        frames = video_to_imgs(msg.video, max_n_frames_per_video=max_n_frames_per_video)
        imgs.extend(frames)

    for filepath in msg.files:
        if is_img(filepath):
            imgs.append(Image.open(filepath))
        elif is_video(filepath):
            frames = video_to_imgs(
                filepath, max_n_frames_per_video=max_n_frames_per_video
            )
            imgs.extend(frames)
        else:
            raise ValueError(f"Unsupported file type: {filepath}")
    return imgs


if __name__ == "__main__":
    test_file_dir = Path(__file__).parent.parent.parent / "test-files"
    filepaths = [test_file_dir / "fish.jpg", test_file_dir / "forest.jpg"]
    for filepath in filepaths:
        assert filepath.exists(), filepath

    msg = Message(
        role="user",
        msg="What is the content of each image?",
        files=filepaths,
    )
    convo = [msg]
    many_convos = [convo] * 20
    model_name = "google/gemma-3-4b-it"
    model = Gemma3vLLM(model_name)
    responses: list[str] = model.complete_batch(many_convos)
    for r in responses:
        print(r)
        print("-" * 100)
