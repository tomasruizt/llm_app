"""
Based on https://github.com/google-gemini/cookbook/blob/main/quickstarts/Video.ipynb
"""

from dataclasses import dataclass
from logging import getLogger
import os
from pathlib import Path
from typing import Literal
import google.generativeai as genai
import time

logger = getLogger(__name__)


@dataclass
class Request:
    media_file: Path
    model_name: Literal["models/gemini-1.5-pro", "models/gemini-1.5-flash"] = (
        "models/gemini-1.5-pro"
    )
    prompt: str = "Describe this video in detail."

    def fetch_media_description(self) -> str:
        return fetch_media_description(self)


def fetch_media_description(req: Request) -> str:
    # TODO: Always delete the video in the end. Perhaps use finally block.
    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

    logger.info("Uploading file: '%s'", req.media_file)
    file = genai.upload_file(path=req.media_file)
    logger.info("Completed upload. URI='%s'", file.uri)

    while file.state.name == "PROCESSING":
        logger.info("Waiting for file to be processed: %s", file.uri)
        time.sleep(1)
        file = genai.get_file(file.name)

    if file.state.name == "FAILED":
        raise ValueError(file.state.name)
    logger.info("File processing complete: %s", file.uri)

    model = genai.GenerativeModel(model_name=req.model_name)
    prompt = req.prompt
    logger.info(
        "Calling the Google API. Prompt='%s', model_name='%s'", prompt, req.model_name
    )
    response = model.generate_content(
        [prompt, file],
        request_options={"timeout": 600},
        generation_config={"temperature": 0.0},
    )
    if len(response.candidates) == 0:
        raise ValueError(
            "No candidates in response. prompt_feedback='%s'" % response.prompt_feedback
        )

    enum = type(response.candidates[0].finish_reason)
    if response.candidates[0].finish_reason == enum.SAFETY:
        raise UnsafeResponseError(safety_ratings=response.candidates[0].safety_ratings)

    genai.delete_file(file.name)
    logger.info("Deleted file: '%s'", file.uri)

    return response.text


class UnsafeResponseError(Exception):
    def __init__(self, safety_ratings: list) -> None:
        super().__init__("The response was blocked by Google due to safety reasons.")
        self.safety_categories = safety_ratings
