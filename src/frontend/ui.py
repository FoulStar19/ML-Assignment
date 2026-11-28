"""Small Streamlit helpers shared by every tab."""
import inspect

import streamlit as st

try:                                   # newer Streamlit uses width="stretch"; older versions use use_container_width
    STRETCH = {"width": "stretch"} if "width" in inspect.signature(st.altair_chart).parameters else {"use_container_width": True}
except Exception:
    STRETCH = {"use_container_width": True}


def tab_strip(options, index, key):
    """A horizontal tab-like selector; its value drives the cards below it."""
    return st.radio("Select", options, index=index, horizontal=True, key=key, label_visibility="collapsed")
