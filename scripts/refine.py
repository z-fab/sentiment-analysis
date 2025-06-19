import argparse
import json
from pathlib import Path
from typing import List

import openai
import polars as pl
from dotenv import load_dotenv
from loguru import logger
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)
from tqdm import tqdm

load_dotenv()
client = openai.OpenAI()

# ────────────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────────────


def parse_args():
    ap = argparse.ArgumentParser(description="Rotular sentimento de reviews usando LLM")
    ap.add_argument(
        "--batch-size", type=int, default=10, help="Tamanho do lote de envio ao LLM"
    )
    ap.add_argument(
        "--checkpoint", type=int, default=100, help="Grava após N previsões novas"
    )
    return ap.parse_args()


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=2, max=15),
    retry=(
        retry_if_exception_type(openai.RateLimitError)
        | retry_if_exception_type(openai.APIConnectionError)
        | retry_if_exception_type(openai.APITimeoutError)
        | retry_if_exception_type(openai.APIError)
    ),
)
def classify_batch(texts: List[str]) -> List[str]:
    joined = "\n".join(
        f"[{i + 1}] {t.replace(chr(10), ' ')}" for i, t in enumerate(texts)
    )
    SYSTEM_PROMPT = (
        "Você é um classificador de sentimento. "
        "Retorne apenas a classificação: Positivo, Neutro ou Negativo para cada review, mantendo a ordem. "
        "Não adicione explicações ou formatação extra. "
        "As reviews são numeradas e devem ser respondidas na mesma ordem, no formato JSON.\n\n"
        "Exemplo de Resposta:\n"
        '{"labels": ["positivo", "neutro", "negativo", "neutro"]}\n'
    )
    USER_PROMPT = (
        "Classifique cada review abaixo como Positivo, Neutro ou Negativo. "
        "Responda no formato JSON, preservando a ordem:\n\n"
        f"{joined}"
    )
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT},
        ],
        temperature=0,
        timeout=30,
    )
    raw = response.choices[0].message.content.strip()
    logger.debug("Resposta do LLM: {}", raw)

    labels = json.loads(raw)["labels"]
    if len(labels) != len(texts):
        raise ValueError("Número de labels retornado não bate com número de textos.")
    return labels


def save_checkpoint(df: pl.DataFrame, output_path: Path) -> None:
    tmp = output_path.with_suffix(".tmp.parquet")
    df.write_parquet(tmp)
    tmp.replace(output_path)


# # ────────────────────────────────────────────────────────────────────────────────
# # Main refine
# # ────────────────────────────────────────────────────────────────────────────────


def main():
    args = parse_args()
    SILVER_DIR = Path(__file__).parent.parent / "data" / "silver"
    GOLD_DIR = Path(__file__).parent.parent / "data" / "gold"

    if (GOLD_DIR / "train_refined.parquet").exists():
        df = pl.read_parquet(GOLD_DIR / "train_refined.parquet")
    else:
        df = pl.read_parquet(SILVER_DIR / "to_refine.parquet")

    if "sentiment" not in df.columns:
        df = df.with_columns(pl.lit(None, dtype=pl.String).alias("sentiment"))

    mask_to_predict = df["sentiment"].is_null()
    df_to_predict = df.filter(mask_to_predict)

    total_missing = df_to_predict.height
    if total_missing == 0:
        logger.info("Nenhuma previsão pendente")
        return

    logger.info(
        f"{total_missing} reviews ainda sem rótulo → enviando em lotes de {args.batch_size}"
    )

    updated = 0
    for start in tqdm(range(0, total_missing, args.batch_size), desc="Batches"):
        end = start + args.batch_size
        batch_df = df_to_predict.slice(start, args.batch_size)
        texts = batch_df["review_text"].to_list()

        labels = classify_batch(texts)

        df_idx = mask_to_predict.arg_true()[start:end]
        df[df_idx, "sentiment"] = pl.Series(labels)

        updated += len(labels)
        if updated % args.checkpoint == 0:
            save_checkpoint(
                df,
                GOLD_DIR / "train_refined.parquet",
            )
            logger.info(f"Checkpoint salvo ({updated} previsões)")

    # Salva versão final
    save_checkpoint(df, GOLD_DIR / "train_refined.parquet")
    logger.success(f"Terminado! {updated} rótulos gerados e salvos em gold")


if __name__ == "__main__":
    main()
