"""CLI entry point for the Invoice Document Processing Agent.

Usage:
    python main.py --serve
    python main.py --serve --port 8080
"""

from __future__ import annotations

import argparse
import sys

from src.utils.logger import get_logger

logger = get_logger(__name__)


def run_server(host: str = "127.0.0.1", port: int = 8000) -> None:
    """Start the FastAPI server."""
    import uvicorn

    uvicorn.run("src.app:app", host=host, port=port, reload=False, log_level="info")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Invoice Document Processing Agent — AI-powered invoice extraction and validation",
    )
    parser.add_argument("--serve", action="store_true", help="Start the FastAPI server")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Server host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Server port (default: 8000)")

    args = parser.parse_args()

    if args.serve:
        run_server(host=args.host, port=args.port)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
