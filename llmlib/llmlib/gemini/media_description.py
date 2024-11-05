"""
Based on https://github.com/google-gemini/cookbook/blob/main/quickstarts/Video.ipynb
"""

from dataclasses import dataclass
from logging import getLogger
import os
from pathlib import Path
from typing import Literal
import google.generativeai as genai
from google.generativeai.types.file_types import File as GoogleFile
from google.generativeai.types import HarmCategory, HarmBlockThreshold
import time

logger = getLogger(__name__)


_pro = "models/gemini-1.5-pro"
_flash = "models/gemini-1.5-flash"
available_models = [_pro, _flash]


@dataclass
class Request:
    media_files: list[Path]
    model_name: Literal[_pro, _flash] = _pro
    prompt: str = "Describe this video in detail."

    def fetch_media_description(self) -> str:
        return fetch_media_description(self)


def fetch_media_description(req: Request) -> str:
    # TODO: Always delete the video in the end. Perhaps use finally block.
    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

    logger.info("Uploading %d file(s)", len(req.media_files))
    files: list[GoogleFile] = [genai.upload_file(path=f) for f in req.media_files]
    logger.info("Completed uploading %d file(s)", len(files))

    for file in files:
        _wait_for_file_processing(file)
    logger.info("File processing complete")

    model = genai.GenerativeModel(model_name=req.model_name)
    prompt = req.prompt
    logger.info(
        "Calling the Google API. Prompt='%s', model_name='%s'",
        shorten_str(prompt),
        req.model_name,
    )
    response = model.generate_content(
        [prompt, *files],
        request_options={"timeout": 600},
        generation_config={"temperature": 0.0},
        safety_settings=_block_nothing(),
    )
    if len(response.candidates) == 0:
        raise ResponseRefusedException(
            "No candidates in response. prompt_feedback='%s'" % response.prompt_feedback
        )

    enum = type(response.candidates[0].finish_reason)
    if response.candidates[0].finish_reason == enum.SAFETY:
        raise UnsafeResponseError(safety_ratings=response.candidates[0].safety_ratings)

    for file in files:
        genai.delete_file(file.name)
        logger.info("Deleted file: '%s'", file.uri)

    return response.text


def _block_nothing() -> dict[HarmCategory, HarmBlockThreshold]:
    return {
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    }


def _wait_for_file_processing(file: GoogleFile) -> None:
    i = 0
    limit = 120  # seconds
    while file.state.name == "PROCESSING":
        if i % 5 == 0:
            logger.info("Waiting for file to be processed: %s", file.uri)
        time.sleep(1)
        i += 1
        file = genai.get_file(file.name)
        if i >= limit:
            raise TimeoutError(
                "File processing took timed out after %ds: %s" % (limit, file.uri)
            )

    if file.state.name == "FAILED":
        raise ValueError(file.state.name)


def shorten_str(s: str) -> str:
    if len(s) > 100:
        return s[:100] + "..."
    return s


class UnsafeResponseError(Exception):
    def __init__(self, safety_ratings: list) -> None:
        super().__init__(
            "The response was blocked by Google due to safety reasons. Categories: %s"
            % safety_ratings
        )
        self.safety_categories = safety_ratings


class ResponseRefusedException(Exception):
    pass
