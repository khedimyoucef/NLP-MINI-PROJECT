import asyncio
import gc
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


class ActivateModelRequest(BaseModel):
    model_name: str

current_model_name = None
model_lock = asyncio.Lock()
base_device = "cuda" if torch.cuda.is_available() else "cpu"
model, tokenizer = None, None
default_max_new_tokens = int(os.getenv("MAX_NEW_TOKENS", "300"))

@app.on_event("startup")
async def startup_event():
    global model, tokenizer, device
    available = discover_models()
    if not available:
        print("WARNING: No models found in 'models/' directory.")
        print("Download at least one model to use the app. See the project report for instructions.")
        return

    model_dir_env = os.getenv("MODEL_DIR")
    if model_dir_env:
        default_model = Path(model_dir_env).name
    else:
        default_model = DEFAULT_MODEL_NAME

    if default_model not in available:
        default_model = available[0]
        print(f"Default model '{DEFAULT_MODEL_NAME}' not found, using: {default_model}")

    try:
        await activate_model(default_model)
    except Exception as e:
        print(f"WARNING: Could not load model '{default_model}': {e}")
        print("The app will start. Select and activate a model from the UI.")


@app.get("/")
async def root():
    if WEB_DIR.exists():
        return FileResponse(WEB_DIR / "index.html")
    return {"status": "ok", "message": "UI not found; set WEB_DIR or open /webapp."}

@app.get("/models")
async def list_models():
    models = discover_models()
    return {"models": models, "default": current_model_name}

@app.get("/current-model")
async def current_model_info():
    return {
        "model_name": current_model_name,
        "model_loaded": model is not None,
        "device": str(device) if device else None,
        "tokenizer_type": "llama_cpp" if tokenizer == "llama_cpp" else "transformers"
    }


@app.post("/activate-model")
async def activate_model_endpoint(req: ActivateModelRequest):
    if req.model_name not in discover_models():
        raise HTTPException(status_code=400, detail=f"Unknown model: {req.model_name}")
    await activate_model(req.model_name)
    return await current_model_info()

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
    max_new_tokens = resolve_max_new_tokens(req.max_new_tokens)

    if tokenizer == "llama_cpp":
        stream = model.create_chat_completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_new_tokens,
            stream=True,
            temperature=0.85,
            top_p=0.95
        )
        for chunk in stream:
            token = chunk["choices"][0]["delta"].get("content", "")
            if token:
                yield token
        return

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
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

        if model is not None:
            try:
                del model
                del tokenizer
                if base_device == "cuda":
                    torch.cuda.empty_cache()
                gc.collect()
            except Exception as e:
                print(f"Warning: Error during model cleanup: {e}")

        print(f"Loading {model_name} from disk...")
        loaded_model, loaded_tokenizer = load_model(str(model_dir), device=base_device)

        model = loaded_model
        tokenizer = loaded_tokenizer
        if tokenizer == "llama_cpp":
            device = torch.device("cpu")
        else:
            device = next(model.parameters()).device
        current_model_name = model_name
        print(f"Model {model_name} activated on {device}")
        if device.type == "cpu":
            threads = int(os.getenv("TORCH_THREADS", "0"))
            if threads > 0:
                torch.set_num_threads(threads)
