from functools import lru_cache

from core.vectordb import COLLECTION


class ReviewRepository:
    def __init__(self):
        self.collection = COLLECTION

    def get_all_products_id(self) -> list[str]:
        """
        Retrieves a list of all unique product IDs from the review collection.

        Returns:
            list[str]: A list of unique product IDs.
        """
        products = self.collection.get(include=["metadatas"])
        if not products or "metadatas" not in products:
            return []

        return list({meta["product_id"] for meta in products.get("metadatas", []) if "product_id" in meta})

    def has_product_reviews(self, product_id: str) -> bool:
        """
        Checks if there are any reviews associated with a given product.

        Args:
            product_id (str): The unique identifier of the product.

        Returns:
            bool: True if there is at least one review for the specified product, False otherwise.
        """
        reviews = self.collection.get(where={"product_id": product_id})
        return bool(reviews.get("documents", []))

    def get_reviews_by_product_id(self, product_id: str) -> list[dict]:
        """
        Retrieve all reviews associated with a specific product ID.

        Args:
            product_id (str): The unique identifier of the product for which to fetch reviews.

        Returns:
            list[dict]: A list of dictionaries, each containing a 'document' (the review content)
                        and its corresponding 'metadata'.
        """
        reviews = self.collection.get(where={"product_id": product_id}, include=["documents", "metadatas"])
        return [{"document": doc, "metadata": meta} for doc, meta in zip(reviews.get("documents", []), reviews.get("metadatas", []))]

    def get_product_sentiments(self, product_id: str) -> list[str]:
        """
        Retrieves the sentiment labels for all reviews associated with a given product.

        Args:
            product_id (str): The unique identifier of the product.

        Returns:
            list[str]: A list of sentiment labels extracted from the product's reviews.
        """
        reviews = self.collection.get(where={"product_id": product_id}, include=["metadatas"])
        return [meta["sentiment_label"] for meta in reviews.get("metadatas", [])]

    def get_latest_reviews_by_sentiment(self, product_id: str, sentiment: str, limit: int = 5):
        """
        Retrieves the latest reviews for a specific product and sentiment.

        Args:
            product_id (str): The unique identifier of the product.
            sentiment (str): The sentiment label to filter reviews (e.g., "positive", "negative").
            limit (int, optional): The maximum number of latest reviews to return. Defaults to 5.

        Returns:
            list: A list of the most recent review documents matching the specified product and sentiment.
        """
        reviews = self.collection.get(
            where={"$and": [{"product_id": product_id}, {"sentiment_label": sentiment}]},
            include=["documents", "metadatas"],
        )

        documents = reviews.get("documents", [])
        metadatas = reviews.get("metadatas", [])

        if not documents:
            return []

        # Ordena pela data de criação para pegar os mais recentes
        sorted_reviews = sorted(
            zip(documents, metadatas),
            key=lambda item: item[1].get("creation_date", ""),
            reverse=True,
        )

        return [doc for doc, _ in sorted_reviews[:limit]]

    def get_embeddings_by_product_id(self, product_id: str):
        """
        Retrieves the embeddings associated with a specific product ID.

        Args:
            product_id (str): The unique identifier of the product for which embeddings are to be retrieved.

        Returns:
            list: A list of embeddings corresponding to the given product ID. Returns an empty list if no embeddings are found.
        """
        embeddings = self.collection.get(where={"product_id": product_id}, include=["embeddings"]).get("embeddings", [])
        return embeddings

    def get_review_by_similarity(self, query_embeddings: list, n_results: int = 3):
        """
        Retrieves the most similar reviews based on the provided query embeddings.

        Args:
            query_embeddings (list): A list of embeddings representing the query to compare against stored reviews.
            n_results (int, optional): The number of most similar reviews to retrieve. Defaults to 3.

        Returns:
            list: A list of documents corresponding to the most similar reviews. Returns an empty list if no documents are found.
        """
        results = self.collection.query(
            query_embeddings=query_embeddings,
            n_results=n_results,
            include=["documents"],
        )
        documents = results.get("documents")
        if not documents:
            return []
        return documents[0]


@lru_cache()
def get_review_repository():
    return ReviewRepository()
