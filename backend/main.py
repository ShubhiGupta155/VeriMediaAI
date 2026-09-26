from fastapi import FastAPI

from backend.routes.analysis import router as analysis_router

app = FastAPI(title="VeriMedia AI API", version="0.1.0")


@app.get("/health")
def health_check():
    return {"status": "ok"}


app.include_router(analysis_router)