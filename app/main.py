import uvicorn
from api.v1.root import router
from fastapi import FastAPI
from mcp_server.sentiment import mcp

http_app = mcp.http_app(path="/")

app = FastAPI(title="API - Sentiment Analysis DP6", version="1.0.0", lifespan=http_app.lifespan)
app.include_router(router, prefix="/api/v1")
app.mount("/mcp", http_app)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001, log_level="info")
