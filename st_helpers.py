from llmlib.base_llm import Message
from llmlib.model_registry import ModelEntry, ModelRegistry


from llmlib.runtime import filled_model_registry
import streamlit as st
from llmlib.bundler import Bundler


def is_video(media_file) -> bool:
    return media_file.name.endswith(".mp4")


def is_image(media_file) -> bool:
    return media_file.name.endswith((".png", ".jpg", ".jpeg", ".gif", ".bmp"))


@st.cache_resource()
def create_model_bundler() -> Bundler:
    return Bundler(registry=filled_model_registry())


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
