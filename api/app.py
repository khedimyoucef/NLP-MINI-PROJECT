import asyncio
import os
from pathlib import Path
from typing import Optional

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from src.model.model import load_model

app = FastAPI()

BASE_DIR = Path(__file__).resolve().parent.parent
WEB_DIR = BASE_DIR / "webapp"
MODEL_ROOT = BASE_DIR / "models"
DEFAULT_MODEL_NAME = "story-gpt2"

if WEB_DIR.exists():
    app.mount("/webapp", StaticFiles(directory=WEB_DIR), name="webapp")

class GenerateRequest(BaseModel):
    theme: str
    age_group: str
    max_new_tokens: Optional[int] = None
    model_name: Optional[str] = None

model_cache = {}
current_model_name = None
model_lock = asyncio.Lock()
base_device = "cuda" if torch.cuda.is_available() else "cpu"
model, tokenizer = None, None
default_max_new_tokens = int(os.getenv("MAX_NEW_TOKENS", "300"))

@app.on_event("startup")
async def startup_event():
    global model, tokenizer, device
    model_dir_env = os.getenv("MODEL_DIR")
    if model_dir_env:
        default_model = model_dir_env
    else:
        default_model = DEFAULT_MODEL_NAME

    available = discover_models()
    if default_model not in available and MODEL_ROOT.exists():
        default_model = DEFAULT_MODEL_NAME

    await activate_model(default_model)


@app.get("/")
async def root():
    if WEB_DIR.exists():
        return FileResponse(WEB_DIR / "index.html")
    return {"status": "ok", "message": "UI not found; set WEB_DIR or open /webapp."}

@app.get("/models")
async def list_models():
    models = discover_models()
    return {"models": models, "default": current_model_name}

@app.post("/generate")
async def generate(req: GenerateRequest):
    model_name = req.model_name or current_model_name or DEFAULT_MODEL_NAME
    if model_name not in discover_models():
        raise HTTPException(status_code=400, detail=f"Unknown model: {model_name}")

    await activate_model(model_name)
    return StreamingResponse(generate_stream(req), media_type="text/plain")


async def generate_stream(req: GenerateRequest):
    prefix = f"Level: {req.age_group} — "
    prompt = prefix + req.theme
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    max_new_tokens = req.max_new_tokens or default_max_new_tokens

    generated = inputs['input_ids']
    prompt_length = generated.shape[1]

    for _ in range(max_new_tokens):
        with torch.no_grad():
            outputs = model(generated)
            next_token_logits = outputs.logits[:, -1, :]
            next_token = torch.multinomial(torch.softmax(next_token_logits / 0.8, dim=-1), 1)
            generated = torch.cat([generated, next_token], dim=1)

        text = tokenizer.decode(generated[0][prompt_length:], skip_special_tokens=True)
        yield text

        if next_token.item() == tokenizer.eos_token_id:
            break

        await asyncio.sleep(0)  # Yield control


def discover_models():
    if not MODEL_ROOT.exists():
        return []
    return sorted([
        p.name for p in MODEL_ROOT.iterdir() if p.is_dir()
    ])


async def activate_model(model_name: str):
    global model, tokenizer, current_model_name, device
    async with model_lock:
        if current_model_name == model_name and model is not None:
            return

        model_dir = MODEL_ROOT / model_name
        if not model_dir.exists():
            raise HTTPException(status_code=404, detail=f"Model directory not found: {model_name}")

        if model_name in model_cache:
            loaded_model, loaded_tokenizer = model_cache[model_name]
        else:
            loaded_model, loaded_tokenizer = load_model(str(model_dir), device=base_device)
            model_cache[model_name] = (loaded_model, loaded_tokenizer)

        model = loaded_model
        tokenizer = loaded_tokenizer
        device = next(model.parameters()).device
        current_model_name = model_name
        if device.type == "cpu":
            threads = int(os.getenv("TORCH_THREADS", "0"))
            if threads > 0:
                torch.set_num_threads(threads)
