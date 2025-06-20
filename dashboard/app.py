import streamlit as st
from pages.chat import render_chat
from pages.dash import render_dashboard

PAGES = [
    st.Page(render_dashboard, title="Dashboard", icon=":material/monitoring:"),  # :contentReference[oaicite:2]{index=2}
    st.Page(render_chat, title="Chat", icon=":material/chat_bubble:"),
]

page = st.navigation(PAGES, position="top")
page.run()
