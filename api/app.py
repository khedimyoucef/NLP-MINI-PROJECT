import os
from pathlib import Path
from typing import Optional

import torch
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from src.model.model import load_model

app = FastAPI()

BASE_DIR = Path(__file__).resolve().parent.parent
WEB_DIR = BASE_DIR / "webapp"
DEFAULT_MODEL_DIR = BASE_DIR / "models" / "story-gpt2"

if WEB_DIR.exists():
    app.mount("/webapp", StaticFiles(directory=WEB_DIR), name="webapp")

class GenerateRequest(BaseModel):
    theme: str
    age_group: str
    max_new_tokens: Optional[int] = None

model, tokenizer = None, None
device = None
default_max_new_tokens = int(os.getenv("MAX_NEW_TOKENS", "300"))

@app.on_event("startup")
async def startup_event():
    global model, tokenizer, device
    model_dir_env = os.getenv("MODEL_DIR")
    if model_dir_env:
        model_id = model_dir_env
    elif DEFAULT_MODEL_DIR.exists():
        model_id = str(DEFAULT_MODEL_DIR)
    else:
        model_id = os.getenv("MODEL_ID", "gpt2")

    model, tokenizer = load_model(model_id)
    device = next(model.parameters()).device
    if device.type == "cpu":
        threads = int(os.getenv("TORCH_THREADS", "0"))
        if threads > 0:
            torch.set_num_threads(threads)


@app.get("/")
async def root():
    if WEB_DIR.exists():
        return FileResponse(WEB_DIR / "index.html")
    return {"status": "ok", "message": "UI not found; set WEB_DIR or open /webapp."}

@app.post("/generate")
async def generate(req: GenerateRequest):
    prefix = f"Level: {req.age_group} — "
    prompt = prefix + req.theme
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    max_new_tokens = req.max_new_tokens or default_max_new_tokens
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.8,
            eos_token_id=tokenizer.eos_token_id,
        )
    text = tokenizer.decode(out[0], skip_special_tokens=True)
    return {"text": text}
