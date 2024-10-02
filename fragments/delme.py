import streamlit as st


def chat_flow():
    st.write("What is your name?")
    input: str = yield
    st.write("Hello:", input)
    st.write("My name is bot!")
    yield


st.title("Streamlit app")
chat = chat_flow()
next(chat)
name: str | None = st.text_input("Enter a string")
if name != "":
    chat.send(name)
