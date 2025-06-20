import asyncio
import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from fastmcp import Client
from google import genai
from google.genai import types

load_dotenv(dotenv_path=Path().resolve() / ".env", override=True)

# ────────────────────────────────────────────────────────────────────────────────
# Config
# ────────────────────────────────────────────────────────────────────────────────

API_URL = os.getenv("API_URL")
MCP_URL = f"{os.getenv('API_URL')}/mcp/"

# ────────────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────────────


async def run_gemini(assistant_placeholder):
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    mcp_client = Client(MCP_URL)
    async with mcp_client:
        formatted_messages = [
            types.Content(role=msg["role"], parts=[types.Part(text=part) for part in msg["parts"]]) for msg in st.session_state.messages
        ]

        response = await client.aio.models.generate_content(
            model="gemini-2.0-flash",
            contents=formatted_messages,
            config=types.GenerateContentConfig(
                tools=[mcp_client.session],
                temperature=0.2,
                max_output_tokens=512,
            ),
        )

        full_reply = response.text if response.text else ""
        assistant_placeholder.write(full_reply)

        st.session_state.messages.append({"role": "model", "parts": [full_reply]})


# ────────────────────────────────────────────────────────────────────────────────
# UI
# ────────────────────────────────────────────────────────────────────────────────


def render_chat():
    st.set_page_config(page_title="Chat", page_icon="💬")

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "user",
                "parts": ["Você é um assistente que responde em português e chama a ferramenta MCP quando necessário."],
            },
            {
                "role": "model",
                "parts": ["Olá! Como posso ajudar você hoje?"],
            },
        ]

    for msg in st.session_state.messages[1:]:
        role_display = "assistant" if msg["role"] == "model" else "user"
        st.chat_message(role_display).write(msg["parts"][0])

    if prompt := st.chat_input("Pergunte algo… ex.: 'Qual o sentimento do produto B08L5VXYZ?'"):
        # Adiciona a mensagem do usuário no histórico
        st.session_state.messages.append({"role": "user", "parts": [prompt]})
        st.chat_message("user").write(prompt)

        # Placeholder para resposta streaming
        assistant_placeholder = st.chat_message("assistant").empty()

        asyncio.run(run_gemini(assistant_placeholder))
