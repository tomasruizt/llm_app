from io import BytesIO
import logging
from PIL import Image
import streamlit as st
from llmlib.runtime import filled_model_registry
from llmlib.model_registry import ModelRegistry
from llmlib.base_llm import Message
from llmlib.bundler import Bundler
from llmlib.bundler_request import BundlerRequest
from st_helpers import (
    create_model_bundler,
    display_warnings,
    is_image,
    is_video,
    render_message,
    render_messages,
    render_gpu_consumption,
)
from login_mask_simple import check_password

fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.INFO, format=fmt)

if not check_password():
    st.stop()

st.set_page_config(page_title="LLM App", layout="wide")

st.title("LLM App")


model_registry: ModelRegistry = filled_model_registry()
model_bundler: Bundler = create_model_bundler()

cs = st.columns([2, 1, 1])
with cs[0]:
    model1_id: str = st.selectbox("Select model", model_registry.all_model_ids())
    display_warnings(model_registry, model1_id)
with cs[1]:
    if "img-key" not in st.session_state:
        st.session_state["img-key"] = 0
    media_file = st.file_uploader(
        "Include an image/video", key=st.session_state["img-key"]
    )

if "messages1" not in st.session_state:
    st.session_state.messages1 = []  # list[Message]


col1, col2, *_ = st.columns([1, 1, 4])
with col1:
    if st.button("Restart chat"):
        st.session_state.messages1 = []  # list[Message]
with col2:
    if st.button("Keep only 1st message"):
        st.session_state.messages1 = st.session_state.messages1[:1]

render_messages(st.session_state.messages1)

with cs[2]:
    render_gpu_consumption()
    if st.button("Clear GPU", help="Some memory might not be freed despite this..."):
        model_bundler.clear_model_on_gpu()

prompt = st.chat_input("Type here")
if prompt is None:
    st.stop()


if media_file is None:
    msg = Message.from_prompt(prompt)
elif is_video(media_file):
    msg = Message(
        role="user",
        msg=prompt,
        video=BytesIO(media_file.read()),
    )
elif is_image(media_file):
    msg = Message(
        role="user",
        msg=prompt,
        img_name=media_file.name,
        img=Image.open(media_file),
    )
else:
    raise ValueError(f"Unsupported file type: {media_file.name}")

if media_file is not None:
    st.session_state["img-key"] += 1

st.session_state.messages1.append(msg)
render_message(msg)

with st.spinner("Initializing model..."):
    model_bundler.set_model_on_gpu(model_id=model1_id)

with st.spinner("Generating response..."):
    req = BundlerRequest(model_id=model1_id, msgs=st.session_state.messages1)
    response = model_bundler.get_response(req)
msg = Message(role="assistant", msg=response)
st.session_state.messages1.append(msg)
render_message(msg)
