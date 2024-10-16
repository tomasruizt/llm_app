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
    video_path: Path
    model_name: Literal["models/gemini-1.5-pro", "models/gemini-1.5-flash"] = (
        "models/gemini-1.5-pro"
    )
    prompt: str = "Describe this video in detail."

    def fetch_video_description(self) -> str:
        return fetch_video_description(self)


def fetch_video_description(req: Request) -> str:
    # TODO: Always delete the video in the end. Perhaps use finally block.
    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

    logger.info("Uploading file: '%s'", req.video_path)
    video_file = genai.upload_file(path=req.video_path)
    logger.info("Completed upload. URI='%s'", video_file.uri)

    while video_file.state.name == "PROCESSING":
        logger.info("Waiting for video to be processed: %s", video_file.uri)
        time.sleep(1)
        video_file = genai.get_file(video_file.name)

    if video_file.state.name == "FAILED":
        raise ValueError(video_file.state.name)
    logger.info("Video processing complete: %s", video_file.uri)

    model = genai.GenerativeModel(model_name=req.model_name)
    prompt = req.prompt
    logger.info(
        "Calling the Google API. Prompt='%s', model_name='%s'", prompt, req.model_name
    )
    response = model.generate_content(
        [prompt, video_file],
        request_options={"timeout": 600},
        generation_config={"temperature": 0.0},
    )
    enum = type(response.candidates[0].finish_reason)
    if response.candidates[0].finish_reason == enum.SAFETY:
        raise UnsafeResponseError(safety_ratings=response.candidates[0].safety_ratings)

    genai.delete_file(video_file.name)
    logger.info("Deleted file: '%s'", video_file.uri)

    return response.text


class UnsafeResponseError(Exception):
    def __init__(self, safety_ratings: list) -> None:
        super().__init__("The response was blocked by Google due to safety reasons.")
        self.safety_categories = safety_ratings


if __name__ == "__main__":
    video_file_name = Path("testvideo.mp4")
    config = Request(video_path=video_file_name)
    description = config.fetch_video_description()
    print(description)
