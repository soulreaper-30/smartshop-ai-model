from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import os
from dotenv import load_dotenv
from predictor import SmartShopPredictor

load_dotenv()
app = FastAPI(title="SmartShop AI Prediction Engine")

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

predictor = SmartShopPredictor()

class ProductInput(BaseModel):
    _id: Optional[str] = None
    name: str
    category: str
    price: float
    quantity: int
    minStockAlert: Optional[int] = 10

class PredictRequest(BaseModel):
    products: List[ProductInput]
    shopId: Optional[str] = None

@app.get("/")
def root():
    return {"status": "SmartShop AI Engine running"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(req: PredictRequest):
    products = [p.dict() for p in req.products]
    predictions = predictor.predict(products)
    return {"predictions": predictions}

@app.get("/festival-alerts")
def festival_alerts():
    return {"alerts": predictor.get_festival_alerts()}

@app.get("/weather-insights")
def weather_insights():
    return {"insights": predictor.get_weather_insights()}
