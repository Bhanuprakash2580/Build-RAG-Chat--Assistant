import logging
from pathlib import Path

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.config import ROOT_DIR
from app.routes.chat import router
from app.utils.errors import AppError, app_error_handler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)

app = FastAPI(
    title="TRUEAILAB RAG Chat Assistant",
    description="Production-style GenAI assistant using Retrieval-Augmented Generation.",
    version="1.0.0",
)

frontend_dir = ROOT_DIR / "frontend"
app.mount("/static", StaticFiles(directory=frontend_dir), name="static")
app.include_router(router)
app.add_exception_handler(AppError, app_error_handler)


@app.exception_handler(RequestValidationError)
async def request_validation_handler(_, exc: RequestValidationError):
    first_error = exc.errors()[0] if exc.errors() else {}
    field = first_error.get("loc", ["field"])[-1]
    return await app_error_handler(_, AppError(f"{field} field is required or invalid", 422))


@app.get("/")
def index() -> FileResponse:
    return FileResponse(Path(frontend_dir / "index.html"))
