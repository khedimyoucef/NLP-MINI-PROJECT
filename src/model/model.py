import torch
import os
import glob
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model(model_name="gpt2", device=None):
    if "gguf" in model_name.lower():
        gguf_files = glob.glob(os.path.join(model_name, "*.gguf"))
        if not gguf_files:
            raise FileNotFoundError(f"No .gguf files found in {model_name}")
        from llama_cpp import Llama
        n_threads = int(os.environ.get("TORCH_THREADS", "4"))
        model = Llama(model_path=gguf_files[0], n_ctx=8192, n_threads=n_threads, verbose=False)
        return model, "llama_cpp"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    return model, tokenizer
