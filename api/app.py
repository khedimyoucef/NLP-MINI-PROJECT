from fastapi import FastAPI
from pydantic import BaseModel
from src.model.model import load_model

app = FastAPI()

class GenerateRequest(BaseModel):
    theme: str
    age_group: str

model, tokenizer = None, None

@app.on_event("startup")
async def startup_event():
    global model, tokenizer
    model, tokenizer = load_model("gpt2")

@app.post("/generate")
async def generate(req: GenerateRequest):
    prefix = f"Level: {req.age_group} — "
    prompt = prefix + req.theme
    inputs = tokenizer(prompt, return_tensors="pt")
    out = model.generate(**inputs, max_new_tokens=150, do_sample=True, temperature=0.8)
    text = tokenizer.decode(out[0], skip_special_tokens=True)
    return {"text": text}
