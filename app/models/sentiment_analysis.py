from pydantic import BaseModel, Field


class SentimentAnalysisResponse(BaseModel):
    sentiment: str = Field(..., description="Sentimento predominante (positivo, negativo ou neutro).")
    summary: str = Field(..., description="Resumo dos reviews gerado por LLM.")
    sentiment_distrib: dict[str, float] = Field(
        ...,
        description="Quantidade relativa de reviews por sentimento (positivo, negativo, neutro).",
    )
    positive_points: list[str] = Field(..., description="Pontos positivos do produto")
    negative_points: list[str] = Field(..., description="Pontos negativos do produto")
    top_reviews: list[str] = Field(..., description="Top 3 reviews mais representativos baseados em embedding.")
