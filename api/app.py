import asyncio
import os
from pathlib import Path
from threading import Thread
from typing import Optional

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from src.model.model import load_model
from transformers import TextIteratorStreamer

app = FastAPI()

BASE_DIR = Path(__file__).resolve().parent.parent
WEB_DIR = BASE_DIR / "webapp"
MODEL_ROOT = BASE_DIR / "models"
DEFAULT_MODEL_NAME = "story-gpt2"
AGE_PREFIXES = {
    "Age3-4": "Level: Age3-4 — Simple words.",
    "Age5-6": "Level: Age5-6 — Short sentences.",
    "Age7-8": "Level: Age7-8 — Moderate vocabulary.",
    "Age9-10": "Level: Age9-10 — Longer sentences.",
    "Age11-12": "Level: Age11-12 — Richer vocabulary.",
}

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
    return StreamingResponse(generate_stream(req, model_name), media_type="text/plain")


def build_prompt(theme: str, age_group: str, model_name: str) -> str:
    prefix = AGE_PREFIXES.get(age_group, f"Level: {age_group}")
    clean_theme = " ".join(theme.split())
    if model_name.startswith("gpt2"):
        return (
            f"{prefix}\n"
            f"Write a clear children's story for {age_group} readers about {clean_theme}. "
            "Use complete sentences with a beginning, middle, and ending.\n"
            "Story:\n"
        )
    return f"{prefix} {clean_theme}"


def resolve_max_new_tokens(requested: Optional[int]) -> int:
    if requested is None:
        return default_max_new_tokens
    return max(1, min(int(requested), 1024))


def generation_kwargs_for(model_name: str, max_new_tokens: int) -> dict:
    if model_name.startswith("gpt2"):
        return {
            "max_new_tokens": max_new_tokens,
            "do_sample": True,
            "temperature": 0.8,
            "top_p": 0.92,
            "top_k": 50,
            "repetition_penalty": 1.2,
            "no_repeat_ngram_size": 3,
            "renormalize_logits": True,
            "pad_token_id": tokenizer.eos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "use_cache": True,
        }
    return {
        "max_new_tokens": max_new_tokens,
        "do_sample": True,
        "temperature": 0.85,
        "top_p": 0.95,
        "top_k": 50,
        "repetition_penalty": 1.1,
        "no_repeat_ngram_size": 3,
        "renormalize_logits": True,
        "pad_token_id": tokenizer.eos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "use_cache": True,
    }


def generate_stream(req: GenerateRequest, model_name: str):
    prompt = build_prompt(req.theme, req.age_group, model_name)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    max_new_tokens = resolve_max_new_tokens(req.max_new_tokens)
    streamer = TextIteratorStreamer(
        tokenizer,
        skip_prompt=True,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )
    generation_kwargs = {
        **inputs,
        **generation_kwargs_for(model_name, max_new_tokens),
        "streamer": streamer,
    }

    generation_error = None

    def run_generation():
        nonlocal generation_error
        try:
            with torch.no_grad():
                model.generate(**generation_kwargs)
        except Exception as exc:  # pragma: no cover - surfaced to the stream caller
            generation_error = exc
            streamer.end()

    worker = Thread(target=run_generation, daemon=True)
    worker.start()

    for chunk in streamer:
        if chunk:
            yield chunk

    worker.join(timeout=1)
    if generation_error is not None:
        raise generation_error


def discover_models():
    if not MODEL_ROOT.exists():
        return []
    names = [p.name for p in MODEL_ROOT.iterdir() if p.is_dir()]
    preferred = current_model_name or DEFAULT_MODEL_NAME
    return sorted(names, key=lambda name: (name != preferred, name))


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
