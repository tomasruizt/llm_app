import os
import subprocess
import tempfile
from llmlib.base_llm import Message
from llmlib.model_registry import ModelEntry, ModelRegistry


from llmlib.runtime import filled_model_registry
from llmlib.whisper import Whisper, WhisperOutput
import streamlit as st
from streamlit.runtime.uploaded_file_manager import UploadedFile
from llmlib.bundler import Bundler


def is_video(media_file) -> bool:
    return media_file.name.endswith(".mp4")


def is_image(media_file) -> bool:
    return media_file.name.endswith((".png", ".jpg", ".jpeg", ".gif", ".bmp"))


@st.cache_resource()
def create_model_bundler() -> Bundler:
    return Bundler(registry=filled_model_registry())


@st.cache_resource(show_spinner="Initializing transcription model (Whisper)...")
def create_whisper() -> Whisper:
    return Whisper()


@st.cache_data(show_spinner="Transcribing video...")
def transcribe_video(media_file: UploadedFile) -> WhisperOutput:
    media_file_extension = "." + media_file.name.split(".")[-1]
    with tempfile.NamedTemporaryFile(
        mode="wb", suffix=media_file_extension, delete=False
    ) as f:
        f.write(media_file.read())
        f.flush()
        filename = f.name
    media_file.seek(0)
    whisper = create_whisper()
    output = whisper.run_pipe(filename, translate=False, return_timestamps=True)
    os.remove(filename)
    return output


def display_warnings(r: ModelRegistry, model_id: str) -> None:
    e1: ModelEntry = r.get_entry(model_id)
    if len(e1.infos) > 0:
        txt = ["* " + e for e in e1.infos]
        st.info("\n".join(txt))
    if len(e1.warnings) > 0:
        txt = ["* " + e for e in e1.warnings]
        st.warning("\n".join(txt))


def render_img(msg: Message):
    st.image(msg.img, caption=msg.img_name, width=400)


def render_video(msg: Message):
    cs = st.columns([1, 4])
    with cs[0]:
        st.video(msg.video)


def render_message(msg: Message):
    with st.chat_message(msg.role):
        if msg.has_image():
            render_img(msg)
        if msg.has_video():
            render_video(msg)
        st.markdown(msg.msg)


def render_messages(msgs: list[Message]) -> None:
    for msg in msgs:
        render_message(msg)


def render_gpu_consumption() -> None:
    output = subprocess.run(
        "nvidia-smi --query-gpu=memory.used --format=csv",
        shell=True,
        capture_output=True,
        text=True,
    )
    memory_used_mb = int(output.stdout.split("\n")[1].replace(" MiB", ""))
    memory_used_gb = memory_used_mb / 1024
    st.metric("Used GPU Memory", f"{memory_used_gb:.2f} GB", delta_color="normal")
    st.button("Update Display")
