import os
import time

import langchain_helper
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel

load_dotenv()

APP_API_KEY = os.getenv("APP_API_KEY")
if not APP_API_KEY:
    raise RuntimeError("APP_API_KEY is not set in .env")

API_KEY_CREDITS = {APP_API_KEY: 5}   # total uses per key
RATE_LIMIT = 2                        # max requests...
WINDOW_SECONDS = 60                   # ...per this many seconds
request_times = {}                    # key -> list of request times

app = FastAPI()

class GenerateRequest(BaseModel):
    cuisine : str

def verify_api_key(x_api_key: str = Header(None)):
    if x_api_key not in API_KEY_CREDITS:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    if API_KEY_CREDITS[x_api_key] <= 0:
        raise HTTPException(status_code=402, detail="No credits left")
    return x_api_key

def check_rate_limit(x_api_key: str = Depends(verify_api_key)):
    now = time.time()
    recent = [t for t in request_times.get(x_api_key, []) if now - t < WINDOW_SECONDS]
    if len(recent) >= RATE_LIMIT:
        raise HTTPException(
            status_code=429,
            detail=f"Too many requests. Max {RATE_LIMIT} per {WINDOW_SECONDS} seconds.",
        )
    recent.append(now)
    request_times[x_api_key] = recent
    return x_api_key

@app.post("/generate")
def generate(req: GenerateRequest, x_api_key: str = Depends(check_rate_limit)):
    try:
        result = langchain_helper.generate_restaurant_name_and_items(req.cuisine)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    
    API_KEY_CREDITS[x_api_key] -= 1
    return {**result, "credit_left": API_KEY_CREDITS[x_api_key]}