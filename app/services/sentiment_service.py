import json
from collections import Counter
from functools import lru_cache

import numpy as np
from core.config import settings
from loguru import logger
from openai import OpenAI
from repositories.review_repository import ReviewRepository, get_review_repository


class SentimentService:
    def __init__(self, review_repo: ReviewRepository):
        self.review_repo = review_repo
        self.openai_client = OpenAI(api_key=settings.OPENAI_API_KEY)

    def _summarize_reviews_with_llm(self, reviews: list[str]) -> str:
        """
        Summarizes a list of product reviews into a single paragraph using an LLM (Large Language Model).

        Args:
            reviews (list[str]): A list of review strings to be summarized.

        Returns:
            str: A single-paragraph summary highlighting the main points from the reviews.
                 If an error occurs during summarization, returns a default error message.
        """
        reviews_text = "\n- ".join(reviews)
        system = (
            "Você é um assistente especializado em resumir reviews de produtos. "
            "Sua tarefa é criar um resumo conciso e informativo dos reviews fornecidos, "
            "destacando os principais pontos mencionados. "
            "Evite incluir opiniões pessoais ou informações irrelevantes. "
            "O resumo deve ser escrito de forma impessoal e objetiva, "
            "sem o uso de primeira pessoa como 'eu', 'meu', ou expressões como 'gostei' ou 'recomendo'. "
            "O resumo deve ser claro e direto, focando apenas nos aspectos do produto em si, "
            "sem mencionar problemas de entrega ou atendimento ao cliente."
            "Caso não haja reviews com informações relevantes sobre o produto, retorne uma mensagem padrão informando que não há informações relacionadas ao produto nos reviews."
        )
        prompt = f"Resuma os seguintes reviews de um produto:\n- {reviews_text}"

        try:
            logger.debug("Gerando resumo dos reviews com LLM.")
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=150,
            )
            return response.choices[0].message.content.strip()
        except Exception:
            logger.exception("Erro ao gerar resumo dos reviews com LLM.")
            return "Não foi possível gerar um resumo dos reviews no momento."

    def _extract_positive_negative_points_with_llm(self, reviews: list[str]) -> dict:
        """
        Extracts the main positive and negative points from a list of product reviews using a Large Language Model (LLM).

        Args:
            reviews (list[str]): A list of review strings to be analyzed.

        Returns:
            dict: A dictionary with two keys:
                - 'pontos_positivos': List of main positive points extracted from the reviews.
                - 'pontos_negativos': List of main negative points extracted from the reviews.
            Returns an empty dictionary if extraction fails.
        """
        reviews_text = "\n- ".join(reviews)
        system = (
            "Você é um assistente especializado em análise de sentimentos de produtos. "
            "Sua tarefa é identificar os principais pontos positivos e negativos dos reviews de um produto."
            "Evite pontos que não sejam relacionados ao produto, como problemas de entrega ou atendimento ao cliente."
            "Concentre-se apenas nos aspectos do produto em si."
            "Retorne no formato JSON com duas chaves: 'pontos_positivos' e 'pontos_negativos' e cada ponto como uma lista"
            "Cada ponto deve ser escrito em terceira pessoa, de forma impessoal e objetiva, evitando o uso de primeira pessoa como 'eu', 'meu', ou expressões como 'gostei' ou 'recomendo'."
            "Exemplo de resposta: "
            '{"pontos_positivos": ["Bom desempenho", "Fácil de usar"], "pontos_negativos": ["Bateria fraca", "Preço alto"]}'
        )
        prompt = f"Identifique os pontos positivos e negativos do produto com base nos reviews abaixo:\n- {reviews_text}"

        try:
            logger.debug("Extraindo pontos positivos e negativos com LLM.")
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                max_tokens=150,
            )
            content = response.choices[0].message.content.strip()
            json_response = json.loads(content)
            return json_response
        except Exception:
            logger.exception("Erro ao extrair pontos positivos e negativos com LLM.")
            return {}

    def _sentiment_summary(self, product_id: str) -> tuple[str, dict]:
        """
        Analyzes and summarizes the sentiment distribution for a given product.

        Args:
            product_id (str): The unique identifier of the product whose reviews will be analyzed.

        Returns:
            tuple[str, dict]: A tuple containing:
                - The predominant sentiment label (str) for the product.
                - A dictionary (dict) mapping each sentiment label (capitalized) to its proportion among all sentiments.
        """
        logger.info(f"Analisando sentimento para o produto {product_id}.")
        sentiments = self.review_repo.get_product_sentiments(product_id)

        sentiment_counts = Counter(sentiments)
        total = len(sentiments)

        predominant_sentiment = sentiment_counts.most_common(1)[0][0]
        sentiment_distrib = {label.capitalize(): round(count / total, 2) for label, count in sentiment_counts.most_common()}

        return predominant_sentiment, sentiment_distrib

    def _positive_negative_points_summary(self, product_id: str) -> tuple[list[str], list[str]]:
        """
        Extracts and summarizes positive and negative review points for a given product.

        Args:
            product_id (str): The unique identifier of the product for which to extract review points.

        Returns:
            tuple[list[str], list[str], str]:
                - A list of positive points extracted from the latest positive reviews.
                - A list of negative points extracted from the latest negative reviews.
                - A summary string generated from the combined latest positive and negative reviews.

        Logs:
            Logs the extraction process for the specified product.
        """
        logger.info(f"Extraindo pontos positivos e negativos para o produto {product_id}.")
        latest_positive_reviews = self.review_repo.get_latest_reviews_by_sentiment(product_id, "positivo", limit=5)
        latest_negative_reviews = self.review_repo.get_latest_reviews_by_sentiment(product_id, "negativo", limit=5)
        cocant_reviews = latest_positive_reviews + latest_negative_reviews
        points = self._extract_positive_negative_points_with_llm(cocant_reviews)

        latest_neutral_reviews = self.review_repo.get_latest_reviews_by_sentiment(product_id, "neutro", limit=5)
        summary = self._summarize_reviews_with_llm(cocant_reviews + latest_neutral_reviews)

        return points["pontos_positivos"], points["pontos_negativos"], summary

    def _top_reviews_summary(self, product_id: str) -> list[str]:
        """
        Generates a summary of the top representative reviews for a given product.

        This method retrieves the embeddings of all reviews associated with the specified product,
        computes the mean embedding to represent the overall sentiment, and then fetches the top
        three reviews most similar to this mean embedding.

        Args:
            product_id (str): The unique identifier of the product for which to summarize reviews.

        Returns:
            list[str]: A list containing the top representative reviews for the product.
        """
        logger.info(f"Buscando os reviews mais representativos para o produto {product_id}.")
        embeddings = self.review_repo.get_embeddings_by_product_id(product_id)
        top_reviews = []

        if embeddings is not None and embeddings.shape[0] > 0:
            mean_embedding = np.mean(embeddings, axis=0)
            top_reviews = self.review_repo.get_review_by_similarity(query_embeddings=[mean_embedding.tolist()], n_results=3)

        return top_reviews

    def analyze_sentiment(self, product_id: str):
        """
        Analyzes the sentiment of reviews for a given product.

        Args:
            product_id (str): The unique identifier of the product to analyze.

        Returns:
            dict: A dictionary containing the following keys:
                - "product_id" (str): The ID of the analyzed product.
                - "predominant_sentiment" (str): The overall predominant sentiment (e.g., positive, negative, neutral).
                - "sentiment_distrib" (dict): Distribution of sentiments among the reviews.
                - "summary" (str): A summary of the reviews' sentiment.
                - "positive_points" (list): List of positive aspects mentioned in the reviews.
                - "negative_points" (list): List of negative aspects mentioned in the reviews.
                - "top_reviews" (list): List of top reviews for the product.
                - "error" (str, optional): Error message if the product is not found or has no reviews.

        Logs:
            Warning if the product is not found or has no reviews.
        """
        if not self.review_repo.has_product_reviews(product_id):
            logger.warning(f"Produto {product_id} não encontrado ou sem reviews.")
            return {"error": "Produto não encontrado ou não possui reviews."}

        predominant_sentiment, sentiment_distrib = self._sentiment_summary(product_id)

        positive_points, negative_points, summary = self._positive_negative_points_summary(product_id)

        top_reviews = self._top_reviews_summary(product_id)

        return {
            "product_id": product_id,
            "predominant_sentiment": predominant_sentiment,
            "sentiment_distrib": sentiment_distrib,
            "summary": summary,
            "positive_points": positive_points,
            "negative_points": negative_points,
            "top_reviews": top_reviews,
        }


@lru_cache()
def get_sentiment_service():
    repo = get_review_repository()
    return SentimentService(review_repo=repo)
