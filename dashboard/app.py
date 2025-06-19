import asyncio

import httpx
import plotly.express as px
import streamlit as st

# ────────────────────────────────────────────────────────────────────────────────
# Config
# ────────────────────────────────────────────────────────────────────────────────

API_URL = "http://localhost:8001"  # backend local
PRODUCTS_ENDPOINT = f"{API_URL}/v1/products"  # opcional
ANALYZE_ENDPOINT = f"{API_URL}/v1/analyze_sentiment"  # POST ou GET


# ────────────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────────────


async def fetch_products() -> list[dict]:
    async with httpx.AsyncClient() as client:
        resp = await client.get(PRODUCTS_ENDPOINT, timeout=10)
        resp.raise_for_status()
        return resp.json()


async def analyze_product(product_id: int) -> dict:
    async with httpx.AsyncClient() as client:
        resp = await client.get(
            ANALYZE_ENDPOINT,
            params={"product_id": product_id},
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()


def build_donut(distribution: dict) -> None:
    labels = list(distribution.keys())
    values = list(distribution.values())
    fig = px.pie(
        names=labels,
        values=values,
        hole=0.5,
        title="Distribuição de Sentimentos",
    )
    st.plotly_chart(fig, use_container_width=True)


# ────────────────────────────────────────────────────────────────────────────────
# UI
# ────────────────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="Análise de Sentimento", layout="wide")
st.title("Dashboard de Sentimento")


@st.cache_data
def get_products_sync():
    return asyncio.run(fetch_products())


products = get_products_sync()
product_options = {f"ID {product_id}": product_id for product_id in products["products"]}

st.sidebar.header("Configurações")
with st.sidebar:
    st.markdown("### Sobre")
    st.markdown(
        "Este dashboard permite analisar o sentimento de reviews de produtos. Selecione um produto e clique em 'Analisar' para ver os resultados."
    )
    selected_label = st.selectbox("Selecione o Produto", list(product_options.keys()))
    run_btn = st.button("Analisar", use_container_width=True)
    random_btn = st.button("Estou com Sorte", use_container_width=True)

if random_btn:
    import random

    selected_label = random.choice(list(product_options.keys()))
    product_id = selected_label

if run_btn or random_btn:
    product_id = product_options[selected_label]
    with st.spinner("Consultando API…"):
        data = asyncio.run(analyze_product(product_id))

    st.markdown(
        f"<div style='background: #eee; padding: 15px 20px; margin-bottom: 20px; border: 1px solid #ddd; border-radius:4px'>{data['summary']}</div>",
        unsafe_allow_html=True,
    )

    col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
    sentiment_distrib_text = {label: f"{round(dist * 100, 1)}%" for label, dist in data["sentiment_distrib"].items()}
    col1.metric("Sentimento Predominante", data["sentiment"].capitalize(), border=True)
    col2.metric("👍🏻 Positivo", sentiment_distrib_text.get("Positivo", "0%"), border=True)
    col3.metric("😐 Neutro", sentiment_distrib_text.get("Neutro", "0%"), border=True)
    col4.metric("👎🏻 Negativo", sentiment_distrib_text.get("Negativo", "0%"), border=True)

    build_donut(data["sentiment_distrib"])
    st.markdown("---")
    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("Pontos Positivos")
        if len(data["positive_points"]) == 0:
            st.markdown("<p style='color: #aaa; font-style: italic'>Não há pontos positivos para este produto.</p>", unsafe_allow_html=True)
        else:
            st.markdown("•  " + "<br/>•  ".join(data["positive_points"]), unsafe_allow_html=True)

    with col2:
        st.subheader("Pontos Negativos")
        if len(data["negative_points"]) == 0:
            st.markdown("<p style='color: #aaa; font-style: italic'>Não há pontos negativos para este produto.</p>", unsafe_allow_html=True)
        else:
            st.markdown("•  " + "<br/>•  ".join(data["negative_points"]), unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("Top 3 Reviews")
    for i, review in enumerate(data["top_reviews"], 1):
        st.markdown(f"**{i}.** {review}")
