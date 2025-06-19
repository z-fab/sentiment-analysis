from fastapi import APIRouter, Depends, HTTPException
from loguru import logger
from models.sentiment_analysis import SentimentAnalysisResponse
from repositories.review_repository import ReviewRepository, get_review_repository
from services.sentiment_service import SentimentService, get_sentiment_service

router = APIRouter()


@router.get("/healthcheck")
async def healthcheck():
    """Endpoint de verificação de saúde da API."""
    return {"status": "ok"}


@router.get("/analyze_sentiment", response_model=SentimentAnalysisResponse)
async def analyze_sentiment(product_id: str, service: SentimentService = Depends(get_sentiment_service)):
    """
    Analisa o sentimento dos reviews para um determinado produto.

    Recebe um `product_id` e retorna:
    - Sentimento predominante para o produto.
    - Um resumo dos reviews para aquele produto.
    - Os pontos positivos e negativos do produto.
    - Os 3 reviews mais representativos.
    """
    result = service.analyze_sentiment(product_id)
    if "error" in result:
        logger.error(f"Erro ao analisar sentimento para o produto {product_id}: {result['error']}")
        raise HTTPException(status_code=404, detail=result["error"])
    return {
        "sentiment": result["predominant_sentiment"],
        "summary": result["summary"],
        "sentiment_distrib": result["sentiment_distrib"],
        "positive_points": result["positive_points"],
        "negative_points": result["negative_points"],
        "top_reviews": result["top_reviews"],
    }


@router.get("/products")
async def get_products(repository: ReviewRepository = Depends(get_review_repository)):
    """
    Retorna a lista de produtos disponíveis para análise de sentimento.
    """

    products = repository.get_all_products_id()
    if not products:
        raise HTTPException(status_code=404, detail="Nenhum produto encontrado.")
    return {"products": products}
