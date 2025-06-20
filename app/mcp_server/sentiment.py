from fastmcp import FastMCP
from loguru import logger
from services.sentiment_service import get_sentiment_service

mcp = FastMCP("Sentiment-MCP")


@mcp.tool
def product_sentiment(product_id: str) -> dict:
    """
    Retorna distribuição de sentimentos, ponto positivo/negativo dominante
    e top-3 reviews de um product_id.
    """
    service = get_sentiment_service()
    result = service.analyze_sentiment(product_id)
    if "error" in result:
        logger.error(f"Erro ao analisar sentimento para o produto {product_id}: {result['error']}")
        return f"Erro ao analisar sentimento para o produto {product_id}: {result['error']}"

    return {
        "sentiment": result["predominant_sentiment"],
        "summary": result["summary"],
        "sentiment_distrib": result["sentiment_distrib"],
        "positive_points": result["positive_points"],
        "negative_points": result["negative_points"],
        "top_reviews": result["top_reviews"],
    }
