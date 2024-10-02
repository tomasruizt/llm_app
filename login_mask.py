"""
WARNING: This file is duplicated in the projects: llm-app, tiktok. Make sure changes are reflected in all projects!

Copied from https://docs.streamlit.io/knowledge-base/deploy/authentication-without-sso
"""

from functools import cache
import hmac
import logging
import os
import streamlit as st


logger = logging.getLogger(__name__)


def check_password() -> bool:
    """Return `True` if the user is allowed to access the app, `False` otherwise."""
    skip_pw: bool = os.environ.get("USE_STREAMLIT_PASSWORD", "true").lower() == "false"
    if skip_pw:
        log_password_check_skipped()
        return True

    """Returns `True` if the user had a correct password."""

    def login_form():
        """Form with widgets to collect user information"""
        with st.form("Credentials"):
            st.text_input("Username", key="username")
            st.text_input("Password", type="password", key="password")
            st.form_submit_button("Log in", on_click=password_entered)

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if st.session_state["username"] in st.secrets[
            "passwords"
        ] and hmac.compare_digest(
            st.session_state["password"],
            st.secrets.passwords[st.session_state["username"]],
        ):
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # Don't store the username or password.
            del st.session_state["username"]
        else:
            st.session_state["password_correct"] = False

    # Return True if the username + password is validated.
    if st.session_state.get("password_correct", False):
        return True

    # Show inputs for username + password.
    login_form()
    if "password_correct" in st.session_state:
        st.error("😕 User not known or password incorrect")
    return False


@cache  # Print only once per session
def log_password_check_skipped():
    logger.info("Skipping password check because USE_STREAMLIT_PASSWORD=false.")
