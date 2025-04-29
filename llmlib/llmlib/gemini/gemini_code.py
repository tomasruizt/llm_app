"""
Based on https://cloud.google.com/vertex-ai/generative-ai/docs/multimodal/video-understanding
"""

import pandas as pd
from dataclasses import dataclass
from io import BytesIO
import json
from logging import getLogger
from pathlib import Path
from typing import Any, TypeVar
from google.cloud import storage
from google.cloud.storage import transfer_manager
from google.genai.types import (
    Part,
    Content,
    HarmCategory,
    HarmBlockThreshold,
    GenerateContentResponse,
    HttpOptions,
    CreateCachedContentConfig,
    CachedContent,
)
import cv2
import os
from google import genai
import requests
from tqdm import tqdm
import google
from strenum import StrEnum
from ..base_llm import LLM, Message, validate_only_first_message_has_files
from ..error_handling import notify_bugsnag

logger = getLogger(__name__)

project_id = "css-lehrbereich-schwemmer"  # (ToxicAInment) from google cloud console
default_location = "europe-west1"  # https://cloud.google.com/about/locations#europe


class Buckets:
    temp = "css-temp-bucket-for-vertex"
    output = "css-vertex-output"


def storage_uri(bucket: str, blob_name: str) -> str:
    """blob_name starts without a slash"""
    return "gs://%s/%s" % (bucket, blob_name)


class GeminiModels(StrEnum):
    """
    The 3 trailing digits indicate the stable version
    https://cloud.google.com/vertex-ai/generative-ai/docs/learn/model-versions#stable-version

    Context caching is supported only for Gemini 1.5 Pro and Flash
    https://cloud.google.com/vertex-ai/generative-ai/docs/context-cache/context-cache-overview#supported_models
    """

    gemini_25_pro = "gemini-2.5-pro-preview-03-25"
    gemini_20_flash = "gemini-2.0-flash-001"
    gemini_20_flash_lite = "gemini-2.0-flash-lite-001"
    gemini_15_pro = "gemini-1.5-pro"
    gemini_15_flash = "gemini-1.5-flash-002"


available_models = list(GeminiModels)


@dataclass
class MultiTurnRequest:
    messages: list[Message]
    model_name: GeminiModels = GeminiModels.gemini_15_pro
    max_output_tokens: int = 1000
    safety_filter_threshold: HarmBlockThreshold = HarmBlockThreshold.BLOCK_NONE
    delete_files_after_use: bool = True
    use_context_caching: bool = False
    location: str = default_location

    def fetch_media_description(self) -> str:
        return _execute_multi_turn_req(self)


@notify_bugsnag
def _execute_multi_turn_req(req: MultiTurnRequest) -> str:
    # Validation: Only the first message can have file(s)
    validate_only_first_message_has_files(req.messages)

    # Prepare Inputs. Use context caching for media
    client = create_client(location=req.location)
    contents = [convert_to_gemini_format(msg) for msg in req.messages]

    files: list[Path] = filepaths(msg=req.messages[0])
    use_caching = req.use_context_caching and is_long_enough_to_cache(files)
    if use_caching:
        # Assume caching was done before
        cached_content, success = get_cached_content(client, req.model_name, files)
        blobs = []
        if not success:
            cached_content, blobs = cache_content(client, req.model_name, files)
    else:  # Add files to the content
        blobs = upload_files(files=files)
        contents = [*blobs_to_parts(blobs), *contents]
        cached_content = None

    # Call Gemini
    response: GenerateContentResponse = _call_gemini(
        client, req, contents, cached_content
    )

    # Cleanup
    if req.delete_files_after_use:
        delete_blobs(blobs)
    return response.text


def is_long_enough_to_cache(paths: list[Path]) -> bool:
    """
    2025-03-13
    * Minimum tokens to cache is 32,768: https://cloud.google.com/vertex-ai/generative-ai/docs/context-cache/context-cache-overview
    * Tokens per sec of video are 263: https://ai.google.dev/gemini-api/docs/tokens?lang=python#multimodal-tokens
    * I assume that with images we won't get to 32,768 tokens
    """
    min_video_duration = (32768 / 263) + 1  # to be safe
    for p in paths:
        if p.suffix == ".mp4":
            duration = video_duration_in_sec(p)
            if duration > min_video_duration:
                return True
    return False


def video_duration_in_sec(filename: Path) -> float:
    """Inspired by https://stackoverflow.com/a/61572332/5730291, but directly asking for duration did not work."""
    video = cv2.VideoCapture(filename)
    frame_count = video.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = video.get(cv2.CAP_PROP_FPS)
    duration = frame_count / fps
    return duration


def cache_content(
    client: genai.Client, model_id: str, paths: list[Path], ttl: str = f"{60 * 20}s"
) -> tuple[CachedContent, list[storage.Blob]]:
    """Caches the content on Google as describe here: https://cloud.google.com/vertex-ai/generative-ai/docs/context-cache/context-cache-create"""
    logger.info("Caching content for paths: %s", paths)
    blobs = upload_files(files=paths)
    parts = blobs_to_parts(blobs)
    content = Content(role="user", parts=parts)
    config = CreateCachedContentConfig(
        contents=[content], display_name=cache_id(model_id, paths), ttl=ttl
    )
    cached_content = client.caches.create(model=model_id, config=config)
    return cached_content, blobs


def cache_id(model_id: str, paths: list[Path]) -> str:
    return json.dumps(dict(model=model_id, paths=str(paths)))


def get_cached_content(
    client: genai.Client, model_id: str, paths: list[Path]
) -> tuple[CachedContent, bool]:
    for cache in client.caches.list():
        if cache.display_name == cache_id(model_id, paths):
            logger.info(
                "Found cached content for model_id='%s' and paths='%s'", model_id, paths
            )
            return cache, True
    return None, False


def convert_to_gemini_format(msg: Message) -> tuple[Content, list[storage.Blob]]:
    role_map = dict(user="user", assistant="model")
    role = role_map[msg.role]
    parts = [Part.from_text(text=msg.msg)]
    return Content(role=role, parts=parts)


def _call_gemini(
    client: genai.Client,
    req: MultiTurnRequest,
    contents: list[Part],
    cached_content: CachedContent | None = None,
) -> GenerateContentResponse:
    logger.info("Calling the Google API. model_name='%s'", req.model_name)
    config = {
        "temperature": 0.0,
        "max_output_tokens": req.max_output_tokens,
        "safety_settings": safety_filters(req.safety_filter_threshold),
    }
    if isinstance(cached_content, CachedContent):
        config["cached_content"] = cached_content.name

    response: GenerateContentResponse = client.models.generate_content(
        model=req.model_name, contents=contents, config=config
    )
    logger.info("Token usage: %s", response.usage_metadata.to_json_dict())

    if len(response.candidates) == 0:
        raise ResponseRefusedException(
            "No candidates in response. prompt_feedback='%s'" % response.prompt_feedback
        )
    else:
        logger.info("Finish reason: %s", response.candidates[0].finish_reason)

    enum = type(response.candidates[0].finish_reason)
    if response.candidates[0].finish_reason in {enum.SAFETY, enum.PROHIBITED_CONTENT}:
        raise UnsafeResponseError(safety_ratings=response.candidates[0].safety_ratings)

    return response


def create_client(location: str = default_location):
    logger.info(
        "Creating client for location='%s', project_id='%s'", location, project_id
    )
    return genai.Client(
        http_options=HttpOptions(api_version="v1"),
        vertexai=True,
        project=project_id,
        location=location,
    )


def blobs_to_parts(blobs: list[storage.Blob]) -> list[Part]:
    return [blob_to_part(b) for b in blobs]


def blob_to_part(b: storage.Blob) -> Part:
    file_uri = storage_uri(Buckets.temp, b.name)
    return Part.from_uri(file_uri=file_uri, mime_type=mime_type(b.name))


def delete_blobs(blobs: list[storage.Blob]) -> None:
    if len(blobs) == 0:
        return
    for blob in blobs:
        blob.delete()
    logger.info("Deleted %d blob(s)", len(blobs))


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
    if len(files) <= 3:
        blobs = [_upload_single_file(file, Buckets.temp) for file in files]
    else:
        blobs = _upload_batchof_files(files, bucket_name=Buckets.temp)
    return blobs


def _upload_batchof_files(files: list[Path], bucket_name: str) -> list[storage.Blob]:
    n_processes = int(os.cpu_count() * 0.8)
    logger.info(
        "Uploading %d file(s) in batch to bucket '%s'. Using %d processes",
        len(files),
        bucket_name,
        n_processes,
    )
    bucket = _bucket(name=bucket_name)
    files_str = [str(f) for f in files]
    blobs = [bucket.blob(file.name) for file in files]
    transfer_manager.upload_many(
        file_blob_pairs=zip(files_str, blobs),
        skip_if_exists=True,
        raise_exception=True,
        max_workers=n_processes,
        upload_kwargs={"timeout": 300},
    )
    logger.info("Completed batch upload of %d file(s)", len(files))
    return blobs


def _bucket(name: str) -> storage.Bucket:
    client = storage.Client(project=project_id)
    return client.bucket(name)


def _upload_single_file(
    file: Path, bucket_name: str, blob_name: str | None = None
) -> storage.Blob:
    bucket: storage.Bucket = _bucket(name=bucket_name)
    if blob_name is None:
        blob_name = file.name
    blob = bucket.blob(blob_name)
    if blob.exists():
        logger.info("Blob '%s' already exists. Skipping upload...", blob.name)
        return blob
    logger.info("Uploading file '%s' to bucket '%s'", file.name, bucket_name)
    blob.upload_from_filename(str(file))
    return blob


def safety_filters(
    threshold: HarmBlockThreshold,
) -> list[dict[str, Any]]:
    map = {
        HarmCategory.HARM_CATEGORY_UNSPECIFIED: threshold,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: threshold,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: threshold,
        HarmCategory.HARM_CATEGORY_HARASSMENT: threshold,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: threshold,
        HarmCategory.HARM_CATEGORY_CIVIC_INTEGRITY: threshold,
    }
    return [{"category": cat, "threshold": th} for cat, th in map.items()]


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
    use_context_caching: bool = False
    delete_files_after_use: bool = True
    safety_filter_threshold: HarmBlockThreshold = HarmBlockThreshold.BLOCK_NONE
    location: str = default_location  # https://cloud.google.com/about/locations#europe

    requires_gpu_exclusively = False
    model_ids = available_models

    def complete_msgs(self, msgs: list[Message]) -> str:
        delete_files_after_use = self.delete_files_after_use
        if self.use_context_caching:
            delete_files_after_use = False

        req = MultiTurnRequest(
            location=self.location,
            model_name=self.model_id,
            messages=msgs,
            use_context_caching=self.use_context_caching,
            max_output_tokens=self.max_output_tokens,
            delete_files_after_use=delete_files_after_use,
            safety_filter_threshold=self.safety_filter_threshold,
        )
        return req.fetch_media_description()

    def video_prompt(self, video: Path | BytesIO, prompt: str) -> str:
        req = MultiTurnRequest(
            location=self.location,
            model_name=self.model_id,
            messages=[Message(role="user", msg=prompt, video=video)],
            max_output_tokens=self.max_output_tokens,
            safety_filter_threshold=self.safety_filter_threshold,
        )
        return req.fetch_media_description()

    @classmethod
    def get_warnings(cls) -> list[str]:
        return [
            "While Gemini supports multi-turn, and multi-file chat, we have only implemented single-file and single-turn prompts atm."
        ]

    def submit_batch_job(self, entries: list["BatchEntry"], tgt_dir: Path) -> str:
        name: str = submit_batch_job(
            model_id=self.model_id,
            entries=entries,
            tgt_dir=tgt_dir,
            safety_filter_threshold=self.safety_filter_threshold,
            location=self.location,
        )
        return name


def filepaths(msg: Message) -> list[Path]:
    """Can return 0, 1 or 2 paths"""
    paths: list[Path] = []
    if msg.has_image():
        if isinstance(msg.img, Path):
            paths.append(msg.img)
        else:
            raise PathNeededError()
    if msg.has_video():
        if isinstance(msg.video, Path):
            paths.append(msg.video)
        else:
            raise PathNeededError()
    if msg.files is not None:
        paths.extend(msg.files)
    return paths


def PathNeededError():
    return ValueError(
        "To support caching based on filename, please provide a deterministic filepath."
    )


def submit_batch_job(
    model_id: str,
    entries: list["BatchEntry"],
    tgt_dir: Path,
    safety_filter_threshold: HarmBlockThreshold,
    location: str,
) -> str:
    # Create and dump input jsonl file
    input_rows: list[dict] = [to_batch_row(c, safety_filter_threshold) for c in entries]
    tgt_dir.mkdir(parents=True, exist_ok=True)
    input_jsonl = tgt_dir / "input.jsonl"
    pd.DataFrame(input_rows).to_json(input_jsonl, orient="records", lines=True)

    # Upload Input jsonl file
    batch_name = f"batch/{tgt_dir.name}"
    input_blob_name = f"{batch_name}/input.jsonl"
    _upload_single_file(
        file=input_jsonl, bucket_name=Buckets.output, blob_name=input_blob_name
    )

    # Upload media files
    all_files = [file for e in entries for file in e.files]
    for files in tqdm(chunk(all_files, 2500)):
        upload_files(files)

    # Submit batch job
    output_dir = f"{batch_name}/output"
    input_uri: str = storage_uri(Buckets.output, blob_name=input_blob_name)
    output_uri: str = storage_uri(Buckets.output, blob_name=output_dir)
    excluded_fields = list(set(k for e in entries for k in e.row_data.keys()))
    response = submit_batch(
        model_id=model_id,
        batch_name=batch_name,
        input_uri=input_uri,
        output_uri=output_uri,
        excluded_fields=excluded_fields,
        location=location,
    )
    response.raise_for_status()
    logger.info("Successfully submitted batch prediction job. JSON=%s", response.json())
    return response.json()["name"]


@dataclass
class BatchEntry:
    prompt: str
    files: list[Path]
    """The row_data is not seen by the LLM, just used to identify the row"""
    row_data: dict[str, str]


def to_batch_row(be: BatchEntry, threshold: HarmBlockThreshold) -> dict:
    file_parts = [part_dict(f) for f in be.files]
    return {
        **be.row_data,
        "request": {
            "contents": [{"role": "user", "parts": [*file_parts, {"text": be.prompt}]}],
            # TODO: This line below might need serialization
            "safetySettings": safety_filters(threshold=threshold),
            "generation_config": {"temperature": 0.0},
        },
    }


def part_dict(file: Path) -> dict:
    uri = remote_uri(file)
    return {"fileData": {"fileUri": uri, "mimeType": mime_type(file.name)}}


def remote_uri(file: Path) -> str:
    return storage_uri(bucket=Buckets.temp, blob_name=file.name)


T = TypeVar("T")


def chunk(xs: list[T], n: int) -> list[list[T]]:
    return [xs[i : i + n] for i in range(0, len(xs), n)]


def submit_batch(
    model_id: str,
    batch_name: str,
    input_uri: str,
    output_uri: str,
    excluded_fields: list[str],
    location: str,
) -> requests.Response:
    # From https://stackoverflow.com/a/55804230/5730291
    cred, project = google.auth.default()
    # creds.valid is False, and creds.token is None
    # Need to refresh credentials to populate those
    auth_req = google.auth.transport.requests.Request()
    cred.refresh(auth_req)

    response = requests.post(
        url=f"https://{location}-aiplatform.googleapis.com/v1beta1/projects/{project_id}/locations/{location}/batchPredictionJobs",
        headers={"Authorization": f"Bearer {cred.token}"},
        json={
            "display_name": f"{batch_name}",
            "model": f"publishers/google/models/{model_id}",
            "inputConfig": {
                "instancesFormat": "jsonl",
                "gcsSource": {"uris": input_uri},
            },
            "outputConfig": {
                "predictionsFormat": "jsonl",
                "gcsDestination": {"outputUriPrefix": output_uri},
            },
            "instanceConfig": {"excludedFields": excluded_fields},
        },
    )

    return response
