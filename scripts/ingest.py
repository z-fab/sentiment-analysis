import argparse
import json
import os
import pickle
from pathlib import Path
from typing import Iterator

import chromadb
import polars as pl
from chromadb.config import Settings
from dotenv import load_dotenv
from loguru import logger
from sentence_transformers import SentenceTransformer
from sqlalchemy import Engine, create_engine, text

# ─────────────────────────────────────────────────────────────────────────────
# Configuração do ambiente
# ─────────────────────────────────────────────────────────────────────────────
load_dotenv()
DB_URL = os.getenv("DATABASE_URL", "sqlite:///data/olist.db")
CHROMA_HOST = os.getenv("CHROMA_HOST", "localhost")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8000"))
COLLECTION = os.getenv("COLLECTION", "olist_reviews")
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

DB_BATCH = 2500
EMB_BATCH = 256
MODEL_DICT = {}

with open(Path(__file__).parent.parent / "model" / "sentiment_model.pkl", "rb") as f:
    MODEL_DICT = pickle.load(f)

# ────────────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────────────


def sql_engine() -> Engine:
    return create_engine(DB_URL, connect_args={"check_same_thread": False})


def stream_reviews(engine: Engine) -> Iterator[pl.DataFrame]:
    logger.info("Streaming SQL em blocos de {} linhas", DB_BATCH)
    SQL = text(
        """
            SELECT order_reviews.review_id,
                order_reviews.order_id,
                order_items.product_id,
                order_reviews.review_score,
                order_reviews.review_creation_date AS creation_date,
                order_reviews.review_comment_message AS review_text

            FROM order_reviews
            JOIN order_items ON order_items.order_id = order_reviews.order_id

            WHERE order_reviews.review_comment_message IS NOT NULL
            AND order_reviews.review_comment_message != ''
            AND order_reviews.order_id IN (
                SELECT order_id
                FROM order_items
                GROUP BY order_id
                    HAVING COUNT(DISTINCT product_id) = 1
            )
        """
    )
    with engine.connect().execution_options(stream_results=True) as conn:
        cursor = conn.execution_options(yield_per=DB_BATCH).execute(SQL)
        cols = cursor.keys()
        while rows := cursor.fetchmany(DB_BATCH):
            yield pl.from_dicts([dict(zip(cols, r)) for r in rows])


def filter_new(df: pl.DataFrame, seen_ids: set[str]) -> pl.DataFrame:
    df = df.with_columns(
        pl.struct(["order_id", "product_id"])
        .map_elements(
            lambda s: f"{s['order_id']}_{s['product_id']}", return_dtype=pl.Utf8
        )
        .alias("doc_id")
    )
    return df.filter(~pl.col("doc_id").is_in(seen_ids))


def encode_text(model: SentenceTransformer, texts: list[str]):
    logger.info("Codificando {} textos", len(texts))
    return model.encode(texts, batch_size=64)


def upsert_batch(
    collection,
    df: pl.DataFrame,
    embeddings: list[list[float]],
    sentiment_label: list[str],
    sentiment_proba: list[dict],
):
    try:
        collection.upsert(
            ids=df["doc_id"].to_list(),
            documents=df["review_text"].to_list(),
            embeddings=embeddings.tolist(),
            metadatas=[
                {
                    "product_id": pid,
                    "order_id": oid,
                    "review_score": int(score),
                    "creation_date": creation,
                    "sentiment_label": label,
                    "sentiment_proba": proba,
                }
                for pid, oid, score, creation, label, proba in zip(
                    df["product_id"],
                    df["order_id"],
                    df["review_score"],
                    df["creation_date"],
                    sentiment_label,
                    sentiment_proba,
                )
            ],
        )
    except Exception as e:
        logger.error("Erro ao inserir no vector database: {}", e)
        raise


def existing_ids(collection) -> set[str]:
    seen: set[str] = set()
    offset = 0
    while batch := collection.get(limit=1000, offset=offset):
        ids = batch.get("ids", [])
        if not ids:
            break
        seen.update(ids)
        offset += len(ids)
    logger.info("Coleção já contém {} vetores", len(seen))
    return seen


def predict_sentiment(
    model_dict: dict, embeddings: list[list[float]]
) -> tuple[list[str], list[str]]:
    if "model" not in model_dict or "label_encoder" not in model_dict:
        logger.error("Modelo de Predição ou label encoder não encontrados")
        raise ValueError("Modelo de Predição ou label encoder não encontrados")

    model_sentiment = model_dict["model"]
    le = model_dict["label_encoder"]
    predict = []
    predict_proba = []

    for i, emb in enumerate(embeddings):
        proba = model_sentiment.predict_proba([emb])
        label = le.inverse_transform([proba.argmax()])[0]
        predict.append(str(label))
        predict_proba.append(
            json.dumps(
                {
                    "negative": float(proba[0][0]),
                    "neutral": float(proba[0][1]),
                    "positive": float(proba[0][2]),
                }
            )
        )

    return predict, predict_proba


# # ────────────────────────────────────────────────────────────────────────────────
# # Main ingest
# # ────────────────────────────────────────────────────────────────────────────────


def ingest(full_refresh: bool):
    engine = sql_engine()

    client = chromadb.HttpClient(
        host=CHROMA_HOST, port=CHROMA_PORT, settings=Settings(allow_reset=True)
    )

    if full_refresh:
        try:
            client.delete_collection(COLLECTION)
            logger.warning("Coleção {} dropada (full‑refresh)", COLLECTION)
        except KeyError:
            pass

    collection = client.get_or_create_collection(COLLECTION)
    seen_ids = set() if full_refresh else existing_ids(collection)

    model = SentenceTransformer(MODEL_NAME)
    inserted = 0

    for df in stream_reviews(engine):
        df = filter_new(df, seen_ids)
        if df.is_empty():
            continue

        # Removendo doc_ids duplicados
        df = df.unique(subset=["doc_id"], keep="last")

        # Atualiza cache de seen_ids para próximos lotes
        seen_ids.update(df["doc_id"].to_list())

        logger.info("Processando {} reviews", df.height)
        for i in range(0, df.height, EMB_BATCH):
            sub = df.slice(i, EMB_BATCH)
            vecs = encode_text(model, sub["review_text"].to_list())
            sentiment_label, sentiment_proba = predict_sentiment(MODEL_DICT, vecs)

            upsert_batch(collection, sub, vecs, sentiment_label, sentiment_proba)
            inserted += sub.height
            if inserted % 1000 == 0:
                logger.info("{} vetores inseridos…", inserted)

    logger.success("Ingesta concluída → {} novos vetores", inserted)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Ingestão de reviews no ChromaDB")
    p.add_argument(
        "--full-refresh", action="store_true", help="Recria a coleção inteira"
    )
    args = p.parse_args()
    ingest(args.full_refresh)
