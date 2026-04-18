from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from app.routes.web import router as web_router
from app.routes.api import router as api_router

app = FastAPI(title="Smart Drone Traffic Analyzer", version="1.0.0")

app.mount("/static", StaticFiles(directory="app/static"), name="static")
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")
app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")

app.include_router(web_router)
app.include_router(api_router, prefix="/api")
