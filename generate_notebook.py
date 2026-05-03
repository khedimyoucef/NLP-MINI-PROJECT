import json

def cell_md(src): return {"cell_type": "markdown", "metadata": {}, "source": src}
def cell_code(src): return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": src}

META = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.12.12",
        "codemirror_mode": {"name": "ipython", "version": 3},
        "file_extension": ".py", "mimetype": "text/x-python",
        "nbconvert_exporter": "python", "pygments_lexer": "ipython3"},
    "nbformat": 4, "nbformat_minor": 5
}

# ============================================================
#  SHARED CELLS (reused in both notebooks)
# ============================================================

def make_install_cell():
    return cell_code([
        "!pip uninstall -y torchao torchvision\n",
        "!pip install -q -U git+https://github.com/huggingface/transformers.git peft bitsandbytes accelerate datasets huggingface_hub\n"
    ])

def make_login_cell():
    return cell_code([
        "from huggingface_hub import login\n",
        "os.environ['WANDB_DISABLED'] = 'true'\n",
        "try:\n",
        "    from kaggle_secrets import UserSecretsClient\n",
        "    login(token=UserSecretsClient().get_secret('HF_TOKEN'))\n",
        "    print('Logged in via Kaggle Secrets')\n",
        "except Exception:\n",
        "    if 'HF_TOKEN' in os.environ: login(token=os.environ['HF_TOKEN'])\n",
        "    else: print('WARNING: No HF_TOKEN found!')\n",
    ])

def make_dataset_cell(sample_size):
    return cell_code([
        "from datasets import load_dataset\n",
        "ds = load_dataset('ajibawa-2023/Children-Stories-Collection', split='train', cache_dir=os.path.join(BASE_DIR, 'hf_cache'))\n",
        f"sample_size = {sample_size}\n",
        "ds_sample = ds.shuffle(seed=42).select(range(min(sample_size, len(ds))))\n",
        "PREFIXES = [\n",
        "    'Level: Age3-4 — Simple words.', 'Level: Age5-6 — Short sentences.',\n",
        "    'Level: Age7-8 — Moderate vocabulary.', 'Level: Age9-10 — Longer sentences.',\n",
        "    'Level: Age11-12 — Richer vocabulary.'\n",
        "]\n",
        "def format_gemma(ex, idx):\n",
        "    pfx = PREFIXES[idx % len(PREFIXES)]\n",
        "    return {'text': f'<start_of_turn>user\\n{pfx}\\n{ex.get(\"prompt\",\"\")}<end_of_turn>\\n<start_of_turn>model\\n{ex.get(\"text\",\"\")}<end_of_turn>\\n'}\n",
        "ds_fmt = ds_sample.map(format_gemma, with_indices=True, remove_columns=ds_sample.column_names)\n",
        "print(f'Formatted {len(ds_fmt)} samples')\n",
    ])

def make_unwrap_and_lora_cell():
    return cell_code([
        "from peft import LoraConfig, get_peft_model\n",
        "model.gradient_checkpointing_enable()\n",
        "model.enable_input_require_grads()\n",
        "for n, p in model.named_parameters():\n",
        "    p.requires_grad = False\n",
        "    if p.ndim == 1 and 'norm' in n.lower(): p.data = p.data.to(torch.float32)\n",
        "try:\n",
        "    from transformers.models.gemma4.modeling_gemma4 import Gemma4ClippableLinear\n",
        "    ct = 0\n",
        "    for n, m in list(model.named_modules()):\n",
        "        if isinstance(m, Gemma4ClippableLinear):\n",
        "            parts = n.split('.')\n",
        "            setattr(model.get_submodule('.'.join(parts[:-1])), parts[-1], m.linear)\n",
        "            ct += 1\n",
        "    print(f'Unwrapped {ct} ClippableLinear modules')\n",
        "except (ImportError, AttributeError) as e:\n",
        "    print(f'Unwrap skipped: {e}')\n",
        "lora_config = LoraConfig(r=16, lora_alpha=32,\n",
        "    target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'],\n",
        "    lora_dropout=0.05, bias='none', task_type='CAUSAL_LM')\n",
        "model = get_peft_model(model, lora_config)\n",
        "model.print_trainable_parameters()\n",
    ])

def make_tokenize_cell():
    return cell_code([
        "MAX_SEQ_LEN = 256\n",
        "def tok_fn(ex):\n",
        "    o = tokenizer(ex['text'], truncation=True, max_length=MAX_SEQ_LEN, padding='max_length')\n",
        "    o['labels'] = o['input_ids'].copy()\n",
        "    return o\n",
        "ds_tok = ds_fmt.map(tok_fn, batched=True, remove_columns=['text'])\n",
        "sp = ds_tok.train_test_split(test_size=0.05, seed=42)\n",
        "train_ds, eval_ds = sp['train'], sp['test']\n",
        "print(f'Train: {len(train_ds)}, Eval: {len(eval_ds)}')\n",
    ])

def make_free_mem_cell():
    return cell_code([
        "print('Freeing GPU memory...')\n",
        "del model, trainer\n",
        "torch.cuda.empty_cache()\n",
        "gc.collect()\n",
    ])

def make_merge_cell():
    return cell_code([
        "from transformers import AutoModelForCausalLM\n",
        "from peft import PeftModel\n",
        "print('Loading base model in fp16 on CPU for merge...')\n",
        "base = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float16, device_map='cpu')\n",
        "try:\n",
        "    from transformers.models.gemma4.modeling_gemma4 import Gemma4ClippableLinear\n",
        "    for n, m in list(base.named_modules()):\n",
        "        if isinstance(m, Gemma4ClippableLinear):\n",
        "            parts = n.split('.')\n",
        "            setattr(base.get_submodule('.'.join(parts[:-1])), parts[-1], m.linear)\n",
        "except Exception as e: print(e)\n",
        "merged = PeftModel.from_pretrained(base, MODEL_OUT).merge_and_unload()\n",
        "merged.save_pretrained(MERGED_DIR)\n",
        "tokenizer.save_pretrained(MERGED_DIR)\n",
        "print(f'Merged model saved to {MERGED_DIR}')\n",
        "del base, merged; gc.collect()\n",
        "print('Aggressively cleaning disk space...')\n",
        "os.system('rm -rf ~/.cache/huggingface/hub')\n",
        "os.system(f'rm -rf {MODEL_OUT}')\n",
    ])

def make_llama_cpp_cell():
    return cell_code(["%%bash\ngit clone https://github.com/ggerganov/llama.cpp.git\ncd llama.cpp && cmake -B build && cmake --build build --config Release -j 4\n"])

def make_convert_cell(name_tag):
    return cell_code([
        "%%bash\n",
        "pip install ./llama.cpp/gguf-py\n",
        f"python llama.cpp/convert_hf_to_gguf.py /kaggle/working/gemma_merged_fp16 --outfile /kaggle/working/ft-gemma-{name_tag}-fp16.gguf --outtype f16\n",
        "rm -rf /kaggle/working/gemma_merged_fp16\n",
        f"./llama.cpp/build/bin/llama-quantize /kaggle/working/ft-gemma-{name_tag}-fp16.gguf /kaggle/working/ft-gemma-{name_tag}-Q4_K_M.gguf Q4_K_M\n",
        f"rm -f /kaggle/working/ft-gemma-{name_tag}-fp16.gguf\n",
        f"ls -lh /kaggle/working/ft-gemma-{name_tag}-Q4_K_M.gguf\n",
    ])

def make_upload_cell(name_tag, repo_filename):
    return cell_code([
        "from huggingface_hub import HfApi, login\n",
        "from kaggle_secrets import UserSecretsClient\n",
        "try:\n",
        "    user_secrets = UserSecretsClient()\n",
        "    login(token=user_secrets.get_secret('HF_TOKEN'))\n",
        "except Exception as e:\n",
        "    print('Warning: HF_TOKEN not found in secrets. If upload fails, login manually.', e)\n",
        "api = HfApi()\n",
        f"GGUF = '/kaggle/working/ft-gemma-{name_tag}-Q4_K_M.gguf'\n",
        f"api.upload_file(path_or_fileobj=GGUF, path_in_repo='{repo_filename}', repo_id='khedim/NLP-MINI-PROJECT',\n",
        f"    commit_message='Upload fine-tuned Gemma 4 {name_tag.upper()} Q4_K_M GGUF')\n",
        "print('Upload Complete!')\n",
    ])

# ============================================================
#  NOTEBOOK 1: E2B (safe, guaranteed to work)
# ============================================================
e2b = []
e2b.append(cell_md(["# Fine-Tune Gemma 4 E2B-IT (QLoRA) — Safe Version\n",
    "Small model (~2.3B effective params). Fits easily on a single T4.\n",
    "Final GGUF: ~1.5 GB."]))
e2b.append(make_install_cell())
e2b.append(cell_code([
    "import os, sys, torch, gc\n",
    "os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'\n",
    "BASE_DIR = '/kaggle/working'\n",
    "MODEL_OUT = os.path.join(BASE_DIR, 'gemma_lora_output')\n",
    "MERGED_DIR = os.path.join(BASE_DIR, 'gemma_merged_fp16')\n",
    "os.makedirs(MODEL_OUT, exist_ok=True); os.makedirs(MERGED_DIR, exist_ok=True)\n",
    "model_id = 'google/gemma-4-E2B-it'\n",
    "print(f'GPUs: {torch.cuda.device_count()}')\n",
]))
e2b.append(make_login_cell())
e2b.append(make_dataset_cell(10000))
e2b.append(cell_code([
    "from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig\n",
    "bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type='nf4', bnb_4bit_compute_dtype=torch.float16)\n",
    "tokenizer = AutoTokenizer.from_pretrained(model_id)\n",
    "model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb, device_map='auto')\n",
    "print('Model loaded successfully!')\n",
]))
e2b.append(make_unwrap_and_lora_cell())
e2b.append(make_tokenize_cell())
e2b.append(cell_code([
    "from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling\n",
    "args = TrainingArguments(output_dir=MODEL_OUT, per_device_train_batch_size=1, gradient_accumulation_steps=16,\n",
    "    optim='paged_adamw_8bit', save_steps=200, save_total_limit=2, logging_steps=20,\n",
    "    learning_rate=2e-4, max_grad_norm=0.3, num_train_epochs=1, warmup_steps=10,\n",
    "    lr_scheduler_type='cosine', fp16=True, gradient_checkpointing=True,\n",
    "    eval_strategy='no')  # Eval disabled — 256k vocab logits cause OOM during eval\n",
    "trainer = Trainer(model=model, args=args, train_dataset=train_ds,\n",
    "    data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False))\n",
    "print('Training with batch_size=1, grad_accum=16, gradient_checkpointing=True')\n",
    "# Resume from checkpoint if one exists (e.g. after a crash)\n",
    "import glob\n",
    "ckpts = sorted(glob.glob(os.path.join(MODEL_OUT, 'checkpoint-*')))\n",
    "resume = ckpts[-1] if ckpts else None\n",
    "if resume: print(f'Resuming from {resume}')\n",
    "trainer.train(resume_from_checkpoint=resume)\n",
    "trainer.model.save_pretrained(MODEL_OUT); tokenizer.save_pretrained(MODEL_OUT)\n",
]))
e2b.append(make_free_mem_cell())
e2b.append(make_merge_cell())
e2b.append(make_llama_cpp_cell())
e2b.append(make_convert_cell("e2b"))
e2b.append(make_upload_cell("e2b", "finetuned-gemma-4-e2b-it-Q4_K_M.gguf"))

with open("notebooks/kaggle_gemma_e2b_finetune.ipynb", "w") as f:
    json.dump({"cells": e2b, "metadata": META, **{k: META[k] for k in ["nbformat","nbformat_minor"]}}, f, indent=2)
print("Created: kaggle_gemma_e2b_finetune.ipynb")

# ============================================================
#  NOTEBOOK 2: E4B (optimized for 2xT4, batch_size=1)
# ============================================================
e4b = []
e4b.append(cell_md(["# Fine-Tune Gemma 4 E4B-IT (QLoRA) — 2×T4 Optimized\n",
    "Larger model (~4.5B effective params). Uses model parallelism across 2 T4s.\n",
    "**Key:** batch_size=1 + gradient_accumulation=16 to fit in VRAM.\n",
    "Final GGUF: ~5 GB."]))
e4b.append(make_install_cell())
e4b.append(cell_code([
    "import os, sys, torch, gc\n",
    "os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'\n",
    "BASE_DIR = '/kaggle/working'\n",
    "MODEL_OUT = os.path.join(BASE_DIR, 'gemma_lora_output')\n",
    "MERGED_DIR = os.path.join(BASE_DIR, 'gemma_merged_fp16')\n",
    "os.makedirs(MODEL_OUT, exist_ok=True); os.makedirs(MERGED_DIR, exist_ok=True)\n",
    "model_id = 'google/gemma-4-E4B-it'\n",
    "n_gpu = torch.cuda.device_count()\n",
    "for i in range(n_gpu): print(f'GPU {i}: {torch.cuda.get_device_name(i)} — {torch.cuda.get_device_properties(i).total_mem/1e9:.1f} GB')\n",
    "assert n_gpu >= 2, 'This notebook requires 2 GPUs! Select 2xT4 in Kaggle settings.'\n",
]))
e4b.append(make_login_cell())
e4b.append(make_dataset_cell(8000))
e4b.append(cell_code([
    "from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig\n",
    "bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type='nf4', bnb_4bit_compute_dtype=torch.float16)\n",
    "tokenizer = AutoTokenizer.from_pretrained(model_id)\n",
    "model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb,\n",
    "    device_map='balanced', max_memory={0: '14GiB', 1: '14GiB'})\n",
    "print('Model loaded across both GPUs!')\n",
]))
e4b.append(make_unwrap_and_lora_cell())
e4b.append(make_tokenize_cell())
e4b.append(cell_code([
    "from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling\n",
    "# batch_size=1 is critical to avoid OOM on E4B with 2xT4\n",
    "args = TrainingArguments(output_dir=MODEL_OUT, per_device_train_batch_size=1, gradient_accumulation_steps=16,\n",
    "    optim='paged_adamw_8bit', save_steps=200, save_total_limit=2, logging_steps=20,\n",
    "    learning_rate=2e-4, max_grad_norm=0.3, num_train_epochs=1, warmup_steps=10,\n",
    "    lr_scheduler_type='cosine', fp16=True, gradient_checkpointing=True,\n",
    "    eval_strategy='no')  # Eval disabled — 256k vocab logits cause OOM\n",
    "trainer = Trainer(model=model, args=args, train_dataset=train_ds,\n",
    "    data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False))\n",
    "import glob\n",
    "ckpts = sorted(glob.glob(os.path.join(MODEL_OUT, 'checkpoint-*')))\n",
    "resume = ckpts[-1] if ckpts else None\n",
    "if resume: print(f'Resuming from {resume}')\n",
    "trainer.train(resume_from_checkpoint=resume)\n",
    "trainer.model.save_pretrained(MODEL_OUT); tokenizer.save_pretrained(MODEL_OUT)\n",
]))
e4b.append(make_free_mem_cell())
e4b.append(make_merge_cell())
e4b.append(make_llama_cpp_cell())
e4b.append(make_convert_cell("e4b"))
e4b.append(make_upload_cell("e4b", "finetuned-gemma-4-e4b-it-Q4_K_M.gguf"))

with open("notebooks/kaggle_gemma_e4b_finetune.ipynb", "w") as f:
    json.dump({"cells": e4b, "metadata": META, **{k: META[k] for k in ["nbformat","nbformat_minor"]}}, f, indent=2)
print("Created: kaggle_gemma_e4b_finetune.ipynb")
