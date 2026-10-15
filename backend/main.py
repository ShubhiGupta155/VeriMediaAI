from contextlib import asynccontextmanager

from fastapi import FastAPI

from backend.routes.analysis import router as analysis_router
from database.connection import Base, engine
from database import models  # Registers AnalysisRecord with Base


@asynccontextmanager
async def lifespan(app: FastAPI):
    Base.metadata.create_all(bind=engine)
    yield


app = FastAPI(
    title="VeriMedia AI API",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health")
def health_check():
    return {"status": "ok"}


app.include_router(analysis_router)