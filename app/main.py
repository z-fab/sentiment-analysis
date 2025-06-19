import argparse

import uvicorn
from api.v1.root import router
from fastapi import FastAPI

app = FastAPI(title="API - Sentiment Analysis DP6", version="1.0.0")

app.include_router(router, prefix="/v1")


def args_parser():
    parser = argparse.ArgumentParser(description="App - Sentiment Analysis DP6")
    parser.add_argument(
        "--mode",
        choices=["api", "dashboard", "mcp"],
        required=True,
        help="Escolha qual serviço iniciar: 'api', 'dashboard', or 'mcp'",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = args_parser()
    if args.mode == "api":
        uvicorn.run(app, host="0.0.0.0", port=8001, log_level="info")
    elif args.mode == "dashboard":
        # Coloque aqui o código para iniciar o dashboard
        print("Iniciando o dashboard...")
    elif args.mode == "mcp":
        # Coloque aqui o código para iniciar o MCP server
        print("Iniciando o MCP server...")
