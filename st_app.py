from io import BytesIO
from PIL import Image
import streamlit as st
from llmlib.runtime import filled_model_registry
from llmlib.model_registry import ModelEntry, ModelRegistry
from llmlib.base_llm import Message
from llmlib.bundler import Bundler
from llmlib.bundler_request import BundlerRequest
from st_helpers import is_image, is_video
from login_mask_simple import check_password

if not check_password():
    st.stop()

st.set_page_config(page_title="LLM App", layout="wide")

st.title("LLM App")


model_registry: ModelRegistry = filled_model_registry()


@st.cache_resource()
def create_model_bundler() -> Bundler:
    return Bundler(registry=model_registry)


def display_warnings(r: ModelRegistry, model_id: str) -> None:
    e1: ModelEntry = r.get_entry(model_id)
    if len(e1.warnings) > 0:
        txt = ["* " + e for e in e1.warnings]
        st.warning("\n".join(txt))


cs = st.columns(2)
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
    st.session_state.messages2 = []  # list[Message]

if st.button("Restart chat"):
    st.session_state.messages1 = []  # list[Message]
    st.session_state.messages2 = []  # list[Message]


def render_messages(msgs: list[Message]) -> None:
    for msg in msgs:
        render_message(msg)


def render_message(msg: Message):
    with st.chat_message(msg.role):
        if msg.has_image():
            render_img(msg)
        if msg.has_video():
            render_video(msg)
        st.markdown(msg.msg)


def render_img(msg: Message):
    st.image(msg.img, caption=msg.img_name, width=400)


def render_video(msg: Message):
    cs = st.columns([1, 4])
    with cs[0]:
        st.video(msg.video)


n_cols = 1
cs = st.columns(n_cols)
render_messages(st.session_state.messages1)

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

model_bundler: Bundler = create_model_bundler()

with st.spinner("Initializing model..."):
    model_bundler.set_model_on_gpu(model_id=model1_id)

with st.spinner("Generating response..."):
    req = BundlerRequest(model_id=model1_id, msgs=st.session_state.messages1)
    response = model_bundler.get_response(req)
msg = Message(role="assistant", msg=response)
st.session_state.messages1.append(msg)
render_message(msg)
