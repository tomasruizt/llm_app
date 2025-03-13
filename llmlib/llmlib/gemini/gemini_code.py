"""
Based on https://cloud.google.com/vertex-ai/generative-ai/docs/multimodal/video-understanding
"""

from dataclasses import dataclass
from functools import singledispatchmethod
from io import BytesIO
from logging import getLogger
from pathlib import Path
import tempfile
from google.cloud import storage
from google.cloud.storage import transfer_manager
import proto
from vertexai.generative_models import (
    GenerativeModel,
    Part,
    Content,
    HarmCategory,
    HarmBlockThreshold,
    GenerationResponse,
)
from enum import StrEnum
from ..base_llm import LLM, Message
from ..error_handling import notify_bugsnag

import vertexai

logger = getLogger(__name__)

project_id = "css-lehrbereich"  # from google cloud console
location = "europe-west1"  # https://cloud.google.com/about/locations#europe


class Buckets:
    temp = "css-temp-bucket-for-vertex"
    output = "css-vertex-output"


def storage_uri(bucket: str, blob_name: str) -> str:
    """blob_name starts without a slash"""
    return "gs://%s/%s" % (bucket, blob_name)


class GeminiModels(StrEnum):
    gemini_15_pro = "models/gemini-1.5-pro"
    gemini_20_flash = "models/gemini-2.0-flash"
    gemini_20_flash_lite = "models/gemini-2.0-flash-lite"


available_models = [
    GeminiModels.gemini_15_pro,
    GeminiModels.gemini_20_flash,
    GeminiModels.gemini_20_flash_lite,
]


@dataclass
class SingleTurnRequest:
    media_files: list[Path]
    model_name: GeminiModels = GeminiModels.gemini_15_pro
    prompt: str = "Describe this video in detail."
    max_output_tokens: int = 1000
    safety_filter_threshold: HarmBlockThreshold = HarmBlockThreshold.BLOCK_NONE
    delete_files_after_use: bool = True

    def fetch_media_description(self) -> str:
        return _execute_single_turn_req(self)


@dataclass
class MultiTurnRequest:
    messages: list[Message]
    model_name: GeminiModels = GeminiModels.gemini_15_pro
    max_output_tokens: int = 1000
    safety_filter_threshold: HarmBlockThreshold = HarmBlockThreshold.BLOCK_NONE
    delete_files_after_use: bool = True

    def fetch_media_description(self) -> str:
        return _execute_multi_turn_req(self)


@notify_bugsnag
def _execute_single_turn_req(req: SingleTurnRequest) -> str:
    blobs = upload_files(files=req.media_files)
    contents = [*blobs_to_parts(blobs), req.prompt]
    response: GenerationResponse = _call_gemini(req, contents)
    if req.delete_files_after_use:
        delete_blobs(blobs)
    return response.text


@notify_bugsnag
def _execute_multi_turn_req(req: MultiTurnRequest) -> str:
    # only the first message can have file(s)
    for msg in req.messages[1:]:
        if msg.has_image() or msg.has_video():
            raise ValueError("Only the first message can have file(s)")
    all_blobs = []
    contents = []
    for msg in req.messages:
        content, blobs = convert_to_gemini_format(msg)
        contents.append(content)
        all_blobs.extend(blobs)
    response: GenerationResponse = _call_gemini(req, contents)
    if req.delete_files_after_use:
        delete_blobs(all_blobs)
    return response.text


def convert_to_gemini_format(msg: Message) -> tuple[Content, list[storage.Blob]]:
    role_map = dict(user="user", assistant="model")
    role = role_map[msg.role]
    paths = dump_files_return_paths(msg=msg)
    blobs = upload_files(files=paths)
    parts = [*blobs_to_parts(blobs), Part.from_text(text=msg.msg)]
    return Content(role=role, parts=parts), blobs


def _call_gemini(
    req: SingleTurnRequest | MultiTurnRequest, contents: list[Part]
) -> GenerationResponse:
    init_vertex()
    model = GenerativeModel(req.model_name)
    logger.info("Calling the Google API. model_name='%s'", req.model_name)
    response: GenerationResponse = model.generate_content(
        contents=contents,
        generation_config={
            "temperature": 0.0,
            "max_output_tokens": req.max_output_tokens,
        },
        safety_settings=safety_filter(req.safety_filter_threshold),
    )
    logger.info("Token usage: %s", proto.Message.to_dict(response.usage_metadata))

    if len(response.candidates) == 0:
        raise ResponseRefusedException(
            "No candidates in response. prompt_feedback='%s'" % response.prompt_feedback
        )

    enum = type(response.candidates[0].finish_reason)
    if response.candidates[0].finish_reason in {enum.SAFETY, enum.PROHIBITED_CONTENT}:
        raise UnsafeResponseError(safety_ratings=response.candidates[0].safety_ratings)

    return response


def blobs_to_parts(blobs: list[storage.Blob]) -> list[Part]:
    return [
        Part.from_uri(storage_uri(Buckets.temp, b.name), mime_type=mime_type(b.name))
        for b in blobs
    ]


def delete_blobs(blobs: list[storage.Blob]) -> None:
    if len(blobs) == 0:
        return
    for blob in blobs:
        blob.delete()
    logger.info("Deleted %d blob(s)", len(blobs))


def init_vertex() -> None:
    vertexai.init(project=project_id, location=location)


def mime_type(file_name: str) -> str:
    mapping = {
        ".txt": "text/plain",
        ".jpg": "image/jpeg",
        ".png": "image/png",
        ".flac": "audio/flac",
        ".mp3": "audio/mpeg",
        ".mp4": "video/mp4",
    }
    for ext, mime in mapping.items():
        if file_name.endswith(ext):
            return mime
    raise ValueError(f"Unknown mime type for file: {file_name}")


def upload_files(files: list[Path]) -> list[storage.Blob]:
    if len(files) == 0:
        return []
    logger.info("Uploading %d file(s)", len(files))
    bucket = _bucket(name=Buckets.temp)
    files_str = [str(f) for f in files]
    blobs = [bucket.blob(file.name) for file in files]
    transfer_manager.upload_many(
        file_blob_pairs=zip(files_str, blobs),
        skip_if_exists=True,
        raise_exception=True,
    )
    logger.info("Completed file(s) upload")
    return blobs


def _bucket(name: str) -> storage.Bucket:
    client = storage.Client(project=project_id)
    return client.bucket(name)


def upload_single_file(file: Path, bucket: str, blob_name: str) -> storage.Blob:
    logger.info("Uploading file '%s' to bucket '%s' as '%s'", file, bucket, blob_name)
    bucket: storage.Bucket = _bucket(name=bucket)
    blob = bucket.blob(blob_name)
    if blob.exists():
        logger.info("Blob '%s' already exists. Overwriting it...", blob_name)
    blob.upload_from_filename(str(file))
    return blob


def safety_filter(
    threshold: HarmBlockThreshold,
) -> dict[HarmCategory, HarmBlockThreshold]:
    return {
        HarmCategory.HARM_CATEGORY_UNSPECIFIED: threshold,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: threshold,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: threshold,
        HarmCategory.HARM_CATEGORY_HARASSMENT: threshold,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: threshold,
        HarmCategory.HARM_CATEGORY_CIVIC_INTEGRITY: threshold,
    }


class UnsafeResponseError(Exception):
    def __init__(self, safety_ratings: list) -> None:
        super().__init__(
            "The response was blocked by Google due to safety reasons. Categories: %s"
            % safety_ratings
        )
        self.safety_categories = safety_ratings


class ResponseRefusedException(Exception):
    pass


@dataclass
class GeminiAPI(LLM):
    model_id: str = GeminiModels.gemini_20_flash_lite
    max_output_tokens: int = 1000

    requires_gpu_exclusively = False
    model_ids = available_models

    def complete_msgs(self, msgs: list[Message]) -> str:
        if len(msgs) == 1:
            msg = msgs[0]
            paths = dump_files_return_paths(msg)
            req = SingleTurnRequest(
                model_name=self.model_id, media_files=paths, prompt=msg.msg
            )
        else:
            req = MultiTurnRequest(model_name=self.model_id, messages=msgs)
        return req.fetch_media_description()

    @singledispatchmethod
    def video_prompt(self, video, prompt: str) -> str:
        raise NotImplementedError(f"Unsupported video type: {type(video)}")

    @video_prompt.register
    def _(self, video: Path, prompt: str) -> str:
        req = SingleTurnRequest(
            model_name=self.model_id, media_files=[video], prompt=prompt
        )
        return req.fetch_media_description()

    @video_prompt.register
    def _(self, video: BytesIO, prompt: str) -> str:
        path = tempfile.mktemp(suffix=".mp4")
        with open(path, "wb") as f:
            f.write(video.getvalue())
        return self.video_prompt(Path(path), prompt)

    @classmethod
    def get_warnings(cls) -> list[str]:
        return [
            "While Gemini supports multi-turn, and multi-file chat, we have only implemented single-file and single-turn prompts atm."
        ]


def dump_files_return_paths(msg: Message) -> list[Path]:
    """Can return 0, 1 or 2 paths"""
    paths: list[Path] = []
    if msg.has_image():
        temp_file = tempfile.mktemp(suffix=".jpg")
        msg.img.save(temp_file)
        paths.append(Path(temp_file))
    if msg.has_video():
        temp_file = tempfile.mktemp(suffix=".mp4")
        msg.video.save(temp_file)
        paths.append(Path(temp_file))
    return paths
