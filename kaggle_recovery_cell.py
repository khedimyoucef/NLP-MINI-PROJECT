import os
import gc
import torch
from huggingface_hub import login, HfApi
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from kaggle_secrets import UserSecretsClient

# ==========================================
# 1. SETUP & AUTHENTICATION
# ==========================================
try:
    user_secrets = UserSecretsClient()
    login(token=user_secrets.get_secret("HF_TOKEN"))
except Exception as e:
    print("Warning: HF_TOKEN not found in secrets. If upload fails, login manually.", e)

BASE_DIR = '/kaggle/working'
MODEL_OUT = os.path.join(BASE_DIR, 'gemma_lora_output')
MERGED_DIR = os.path.join(BASE_DIR, 'gemma_merged_fp16')
model_id = 'google/gemma-4-2b-it' # Base model ID

# Check if we survived!
if not os.path.exists(MODEL_OUT):
    print("❌ FATAL: The training output folder 'gemma_lora_output' is completely gone. You will need to restart the notebook from scratch.")
    exit(1)

# ==========================================
# 2. MERGE LORA ADAPTER INTO BASE MODEL
# ==========================================
print('\n--- MERGING LORA WEIGHTS ---')
tokenizer = AutoTokenizer.from_pretrained(model_id)
base = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float16, device_map='cpu')

try:
    from transformers.models.gemma4.modeling_gemma4 import Gemma4ClippableLinear
    for n, m in list(base.named_modules()):
        if isinstance(m, Gemma4ClippableLinear):
            parts = n.split('.')
            setattr(base.get_submodule('.'.join(parts[:-1])), parts[-1], m.linear)
except Exception as e:
    print("Unwrapping error (can be ignored if already unwrapped):", e)

merged = PeftModel.from_pretrained(base, MODEL_OUT).merge_and_unload()
merged.save_pretrained(MERGED_DIR)
tokenizer.save_pretrained(MERGED_DIR)
print(f'Merged model successfully saved to {MERGED_DIR}')

del base, merged
gc.collect()

# ==========================================
# 3. AGGRESSIVELY CLEAR DISK SPACE
# ==========================================
print('\n--- CLEANING DISK FOR CONVERSION ---')
# The merge was successful, we no longer need the raw checkpoints or the base model cache!
os.system('rm -rf ~/.cache/huggingface/hub')
os.system(f'rm -rf {MODEL_OUT}')
os.system(f'rm -f {BASE_DIR}/ft-gemma-e2b-fp16.gguf') # delete any failed gguf attempt
os.system('df -h /kaggle/working')

# ==========================================
# 4. CONVERT TO GGUF & QUANTIZE
# ==========================================
print('\n--- CONVERTING TO GGUF AND QUANTIZING ---')
os.system('pip install ./llama.cpp/gguf-py')

# Convert FP16
print("Converting HF to GGUF (FP16)...")
os.system(f'python llama.cpp/convert_hf_to_gguf.py {MERGED_DIR} --outfile {BASE_DIR}/ft-gemma-e2b-fp16.gguf --outtype f16')

# Delete merged safetensors immediately!
os.system(f'rm -rf {MERGED_DIR}')

# Quantize to Q4_K_M
print("Quantizing to Q4_K_M...")
os.system(f'./llama.cpp/build/bin/llama-quantize {BASE_DIR}/ft-gemma-e2b-fp16.gguf {BASE_DIR}/ft-gemma-e2b-Q4_K_M.gguf Q4_K_M')

# Delete FP16 GGUF
os.system(f'rm -f {BASE_DIR}/ft-gemma-e2b-fp16.gguf')

print('\n--- FINAL FILE ---')
os.system(f'ls -lh {BASE_DIR}/ft-gemma-e2b-Q4_K_M.gguf')

# ==========================================
# 5. UPLOAD TO HUGGING FACE
# ==========================================
print('\n--- UPLOADING TO HUGGING FACE ---')
api = HfApi()
gguf_path = os.path.join(BASE_DIR, 'ft-gemma-e2b-Q4_K_M.gguf')

try:
    api.upload_file(
        path_or_fileobj=gguf_path, 
        path_in_repo='finetuned-gemma-4-e2b-it-Q4_K_M.gguf', 
        repo_id='khedim/NLP-MINI-PROJECT',
        commit_message='Upload fine-tuned Gemma 4 E2B Q4_K_M GGUF'
    )
    print('\n🎉 UPLOAD COMPLETE! THE MIRACLE HAPPENED! 🎉')
except Exception as e:
    print("\n❌ Upload failed:", e)
