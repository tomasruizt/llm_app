"""
Based on https://cloud.google.com/vertex-ai/generative-ai/docs/multimodal/video-understanding
"""

import pandas as pd
from dataclasses import dataclass, field
from io import BytesIO
import json
from logging import getLogger
from pathlib import Path
from typing import Any, Iterable, TypeVar
from concurrent.futures import ThreadPoolExecutor, as_completed
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
    ThinkingConfig,
)
import cv2
import os
from google import genai
import requests
from tqdm import tqdm
import google
from strenum import StrEnum
from ..base_llm import LLM, Message, validate_only_first_message_has_files, LlmReq
from ..error_handling import notify_bugsnag
from pydantic import BaseModel

logger = getLogger(__name__)

project_id = "css-lehrbereich-schwemmer"  # (ToxicAInment) from google cloud console
# On what regions is which model available? https://cloud.google.com/vertex-ai/generative-ai/docs/learn/locations#europe
default_location = "us-central1"  # https://cloud.google.com/about/locations#europe


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

    gemini_25_pro = "gemini-2.5-pro"
    default = gemini_25_flash = "gemini-2.5-flash"
    gemini_20_flash = "gemini-2.0-flash-001"
    gemini_20_flash_lite = "gemini-2.0-flash-lite-001"

    # 2025-05-18: Gemini 1.5 is being deprecated
    # gemini_15_pro = "gemini-1.5-pro"
    # gemini_15_flash = "gemini-1.5-flash-002"


available_models = list(GeminiModels)


@dataclass
class MultiTurnRequest:
    messages: list[Message]
    model_name: GeminiModels = GeminiModels.default
    gen_kwargs: dict[str, Any] = field(default_factory=dict)
    safety_filter_threshold: HarmBlockThreshold = HarmBlockThreshold.BLOCK_NONE
    delete_files_after_use: bool = True
    use_context_caching: bool = False
    location: str = default_location
    json_schema: type[BaseModel] | None = None
    output_dict: bool = False
    include_thoughts: bool = False

    def fetch_media_description(self) -> str | dict:
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
    response, config = _call_gemini(client, req, contents, cached_content)

    # Cleanup
    if req.delete_files_after_use:
        delete_blobs(blobs)

    if not req.output_dict:
        return response.text

    data = {"response": response.text, **config}
    reasonings = [p.text for p in response.candidates[0].content.parts if p.thought]
    if len(reasonings) > 1:
        logger.warning("Found %d reasoning parts. Expected 1.", len(reasonings))
    if len(reasonings) > 0:
        data["reasoning"] = reasonings[0]
    return data


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


def convert_to_gemini_format(msg: Message) -> Content:
    role: str = role_map(msg.role)
    parts = [Part.from_text(text=msg.msg)]
    return Content(role=role, parts=parts)


def role_map(role: str) -> str:
    return dict(user="user", assistant="model")[role]


def _call_gemini(
    client: genai.Client,
    req: MultiTurnRequest,
    contents: list[Part],
    cached_content: CachedContent | None = None,
) -> tuple[GenerateContentResponse, dict]:
    logger.info("Calling the Google API. model_name='%s'", req.model_name)
    default_gen_kwargs = {
        "max_output_tokens": 1000,
        "temperature": 0.0,
        "safety_settings": safety_filters(req.safety_filter_threshold),
    }
    config = default_gen_kwargs | req.gen_kwargs
    if req.json_schema is not None:
        config["response_mime_type"] = "application/json"
        config["response_schema"] = req.json_schema

    if req.include_thoughts:
        config["thinking_config"] = ThinkingConfig(include_thoughts=True)

    if isinstance(cached_content, CachedContent):
        config["cached_content"] = cached_content.name

    response: GenerateContentResponse = client.models.generate_content(
        model=req.model_name, contents=contents, config=config
    )
    token_usage = response.usage_metadata.to_json_dict()
    logger.info("Token usage: %s", token_usage)

    if len(response.candidates) == 0:
        raise ResponseRefusedException(
            "No candidates in response. prompt_feedback='%s'" % response.prompt_feedback
        )

    finish_reason = response.candidates[0].finish_reason
    logger.info("Finish reason: %s", finish_reason)

    enum = type(finish_reason)
    if finish_reason in {enum.SAFETY, enum.PROHIBITED_CONTENT}:
        raise UnsafeResponseError(safety_ratings=response.candidates[0].safety_ratings)
    if finish_reason == enum.MAX_TOKENS:
        raise ValueError("Max tokens reached. Token usage: %s" % repr(token_usage))

    return response, config


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
    model_id: str = GeminiModels.default
    use_context_caching: bool = False
    delete_files_after_use: bool = True
    safety_filter_threshold: HarmBlockThreshold = HarmBlockThreshold.BLOCK_NONE
    location: str = default_location  # https://cloud.google.com/about/locations#europe
    max_n_batching_threads: int = 16
    include_thoughts: bool = False

    requires_gpu_exclusively = False
    model_ids = available_models

    def complete_msgs(
        self,
        msgs: list[Message],
        output_dict: bool = False,
        json_schema: type[BaseModel] | None = None,
        **gen_kwargs,
    ) -> str:
        req = self._multiturn_req(
            msgs=msgs,
            output_dict=output_dict,
            gen_kwargs=gen_kwargs,
            json_schema=json_schema,
        )
        return req.fetch_media_description()

    def video_prompt(self, video: Path | BytesIO, prompt: str) -> str:
        msgs = [Message(role="user", msg=prompt, video=video)]
        return self.complete_msgs(msgs=msgs)

    def _multiturn_req(self, msgs: list[Message], **kwargs) -> MultiTurnRequest:
        delete_files_after_use = self.delete_files_after_use
        if self.use_context_caching:
            delete_files_after_use = False
        req = MultiTurnRequest(
            messages=msgs,
            model_name=self.model_id,
            safety_filter_threshold=self.safety_filter_threshold,
            delete_files_after_use=delete_files_after_use,
            use_context_caching=self.use_context_caching,
            location=self.location,
            include_thoughts=self.include_thoughts,
            **kwargs,
        )
        return req

    @classmethod
    def get_warnings(cls) -> list[str]:
        return [
            "While Gemini supports multi-turn, and multi-file chat, we have only implemented single-file and single-turn prompts atm."
        ]

    def submit_batch_job(self, entries: list[LlmReq], tgt_dir: Path | str) -> str:
        name: str = submit_batch_job(
            model_id=self.model_id,
            entries=entries,
            tgt_dir=tgt_dir,
            safety_filter_threshold=self.safety_filter_threshold,
            location=self.location,
        )
        return name

    def complete_batchof_reqs(self, batch: list[LlmReq]) -> Iterable[dict]:
        if len(batch) == 0:
            return []
        n_threads = min(len(batch), self.max_n_batching_threads)
        with ThreadPoolExecutor(max_workers=n_threads) as executor:
            futures = [
                executor.submit(self._complete_single_llmreq, args)
                for args in enumerate(batch)
            ]
            for future in as_completed(futures):
                yield future.result()

    def _complete_single_llmreq(self, args: tuple[int, LlmReq]) -> dict:
        request_idx, req = args
        try:
            mt_kwargs = {"gen_kwargs": req.gen_kwargs, "output_dict": True}
            mt_req = self._multiturn_req(msgs=req.convo, **mt_kwargs)
            data = mt_req.fetch_media_description()
            data = data | {"success": True, "request_idx": request_idx, **req.metadata}
            return data
        except Exception as e:
            logger.error("Error processing request %d: %s", request_idx, e)
            return {
                "success": False,
                "request_idx": request_idx,
                "error": str(e),
                **req.metadata,
            }


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
    entries: list[LlmReq],
    tgt_dir: Path | str,
    safety_filter_threshold: HarmBlockThreshold,
    location: str,
) -> str:
    # Create and dump input jsonl file
    input_rows: list[dict] = [to_batch_row(c, safety_filter_threshold) for c in entries]
    tgt_dir = Path(tgt_dir)
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
    all_files = [file for e in entries for m in e.convo for file in m.files]
    for files in tqdm(chunk(all_files, 2500)):
        upload_files(files)

    # Submit batch job
    output_dir = f"{batch_name}/output"
    input_uri: str = storage_uri(Buckets.output, blob_name=input_blob_name)
    output_uri: str = storage_uri(Buckets.output, blob_name=output_dir)
    excluded_fields = list(set(k for e in entries for k in e.metadata.keys()))
    response = submit_batch(
        model_id=model_id,
        batch_name=batch_name,
        input_uri=input_uri,
        output_uri=output_uri,
        excluded_fields=excluded_fields,
        location=location,
    )
    if response.status_code != 200:
        raise Exception(
            "Failed to submit batch job. status_code={}, response={}".format(
                response.status_code, response.text
            )
        )
    confirmation_data = response.json()
    logger.info(
        "Successfully submitted batch prediction job. JSON=%s", confirmation_data
    )
    confirmation_file = tgt_dir / "submit_confirmation.json"
    confirmation_file.write_text(json.dumps(confirmation_data, indent=2))
    return confirmation_data["name"]


def to_batch_row(be: LlmReq, threshold: HarmBlockThreshold) -> dict:
    contents = [
        {
            "role": role_map(msg.role),
            "parts": [part_dict(f) for f in msg.files] + [{"text": msg.msg}],
        }
        for msg in be.convo
    ]
    return {
        **be.metadata,
        "request": {
            "contents": contents,
            "safetySettings": safety_filters(threshold=threshold),
            "generation_config": be.gen_kwargs,
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
