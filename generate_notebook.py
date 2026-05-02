import json

def cell_md(source):
    return {"cell_type": "markdown", "metadata": {}, "source": source}

def cell_code(source):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": source}

cells = []

# --- Cell 0: Title ---
cells.append(cell_md([
    "# Kaggle End-to-End: Fine-Tune & Quantize Gemma 4 E4B-IT\n",
    "\n",
    "This notebook performs the complete pipeline:\n",
    "1. Downloads the Children Stories dataset.\n",
    "2. Fine-tunes `google/gemma-4-E4B-it` using QLoRA on 2x T4 GPUs.\n",
    "3. Merges the LoRA adapter back into the base model.\n",
    "4. Uses `llama.cpp` to convert the merged model into a `.gguf` file.\n",
    "5. Quantizes the file into a `Q4_K_M` format (~5GB).\n",
    "6. Uploads the final `.gguf` file to your Hugging Face repo."
]))

# --- Cell 1: Install deps ---
cells.append(cell_code([
    "!pip install -q -U git+https://github.com/huggingface/transformers.git peft bitsandbytes trl accelerate datasets huggingface_hub"
]))

# --- Cell 2: Setup ---
cells.append(cell_code([
    "import os\n",
    "import sys\n",
    "import subprocess\n",
    "import torch\n",
    "import gc\n",
    "\n",
    "print('CUDA available:', torch.cuda.is_available())\n",
    "print('GPU count:', torch.cuda.device_count())\n",
    "for i in range(torch.cuda.device_count()):\n",
    "    print(f'GPU {i}:', torch.cuda.get_device_name(i))\n",
    "\n",
    "BASE_DIR = '/kaggle/working'\n",
    "MODEL_OUT = os.path.join(BASE_DIR, 'gemma_lora_output')\n",
    "MERGED_DIR = os.path.join(BASE_DIR, 'gemma_merged_fp16')\n",
    "os.makedirs(MODEL_OUT, exist_ok=True)\n",
    "os.makedirs(MERGED_DIR, exist_ok=True)"
]))

# --- Cell 3: HF Login ---
cells.append(cell_code([
    "from huggingface_hub import login\n",
    "\n",
    "os.environ['WANDB_DISABLED'] = 'true'\n",
    "\n",
    "try:\n",
    "    from kaggle_secrets import UserSecretsClient\n",
    "    user_secrets = UserSecretsClient()\n",
    "    hf_token = user_secrets.get_secret('HF_TOKEN')\n",
    "    login(token=hf_token)\n",
    "    print('Logged in with HF_TOKEN from Kaggle Secrets')\n",
    "except Exception as e:\n",
    "    if 'HF_TOKEN' in os.environ:\n",
    "        login(token=os.environ['HF_TOKEN'])\n",
    "    else:\n",
    "        print('WARNING: No HF_TOKEN found!')\n",
]))

# --- Cell 4: Load & format dataset ---
cells.append(cell_code([
    "from datasets import load_dataset\n",
    "\n",
    "ds = load_dataset('ajibawa-2023/Children-Stories-Collection', split='train', cache_dir=os.path.join(BASE_DIR, 'hf_cache'))\n",
    "print('Total Rows:', len(ds))\n",
    "\n",
    "sample_size = 8000\n",
    "ds_sample = ds.shuffle(seed=42).select(range(min(sample_size, len(ds))))\n",
    "\n",
    "PREFIXES = [\n",
    "    'Level: Age3-4 — Simple words.',\n",
    "    'Level: Age5-6 — Short sentences.',\n",
    "    'Level: Age7-8 — Moderate vocabulary.',\n",
    "    'Level: Age9-10 — Longer sentences.',\n",
    "    'Level: Age11-12 — Richer vocabulary.'\n",
    "]\n",
    "\n",
    "def format_gemma_instruct(example, idx):\n",
    "    prefix = PREFIXES[idx % len(PREFIXES)]\n",
    "    prompt = example.get('prompt', '')\n",
    "    story = example.get('text', '') or example.get('story', '')\n",
    "    text = f'<start_of_turn>user\\n{prefix}\\n{prompt}<end_of_turn>\\n<start_of_turn>model\\n{story}<end_of_turn>\\n'\n",
    "    return {'text': text}\n",
    "\n",
    "ds_formatted = ds_sample.map(format_gemma_instruct, with_indices=True, remove_columns=ds_sample.column_names)\n",
    "print('Sample:', ds_formatted[0]['text'][:300])\n",
]))

# --- Cell 5: Load model in 4-bit ---
cells.append(cell_code([
    "from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig\n",
    "\n",
    "model_id = 'google/gemma-4-E4B-it'\n",
    "\n",
    "bnb_config = BitsAndBytesConfig(\n",
    "    load_in_4bit=True,\n",
    "    bnb_4bit_use_double_quant=True,\n",
    "    bnb_4bit_quant_type='nf4',\n",
    "    bnb_4bit_compute_dtype=torch.float16\n",
    ")\n",
    "\n",
    "tokenizer = AutoTokenizer.from_pretrained(model_id)\n",
    "model = AutoModelForCausalLM.from_pretrained(\n",
    "    model_id,\n",
    "    quantization_config=bnb_config,\n",
    "    device_map='balanced',\n",
    "    max_memory={0: '12GB', 1: '12GB'}\n",
    ")\n",
]))

# --- Cell 6: Prepare LoRA ---
cells.append(cell_code([
    "from peft import LoraConfig, get_peft_model\n",
    "\n",
    "model.gradient_checkpointing_enable()\n",
    "model.enable_input_require_grads()\n",
    "\n",
    "# Manually prepare for kbit training without upcasting massive embeddings\n",
    "for name, param in model.named_parameters():\n",
    "    param.requires_grad = False\n",
    "    if param.ndim == 1 and 'norm' in name.lower():\n",
    "        param.data = param.data.to(torch.float32)\n",
    "\n",
    "# Gemma 4 wraps its Linear layers in Gemma4ClippableLinear.\n",
    "# PEFT does not recognize this wrapper, so we unwrap them to expose\n",
    "# the inner Linear4bit modules that PEFT knows how to inject LoRA into.\n",
    "try:\n",
    "    from transformers.models.gemma4.modeling_gemma4 import Gemma4ClippableLinear\n",
    "    for name, module in list(model.named_modules()):\n",
    "        if isinstance(module, Gemma4ClippableLinear):\n",
    "            parts = name.split('.')\n",
    "            parent = model.get_submodule('.'.join(parts[:-1]))\n",
    "            setattr(parent, parts[-1], module.linear)\n",
    "    print('Unwrapped Gemma4ClippableLinear -> Linear4bit for PEFT compatibility')\n",
    "except ImportError:\n",
    "    print('No Gemma4ClippableLinear found, skipping unwrap')\n",
    "\n",
    "lora_config = LoraConfig(\n",
    "    r=16,\n",
    "    lora_alpha=32,\n",
    "    target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],\n",
    "    lora_dropout=0.05,\n",
    "    bias='none',\n",
    "    task_type='CAUSAL_LM'\n",
    ")\n",
    "\n",
    "model = get_peft_model(model, lora_config)\n",
    "model.print_trainable_parameters()\n",
]))

# --- Cell 7: Tokenize dataset ---
cells.append(cell_code([
    "# Tokenize the formatted dataset for causal LM training\n",
    "MAX_SEQ_LEN = 512\n",
    "\n",
    "def tokenize_fn(examples):\n",
    "    out = tokenizer(\n",
    "        examples['text'],\n",
    "        truncation=True,\n",
    "        max_length=MAX_SEQ_LEN,\n",
    "        padding='max_length',\n",
    "    )\n",
    "    out['labels'] = out['input_ids'].copy()\n",
    "    return out\n",
    "\n",
    "ds_tokenized = ds_formatted.map(tokenize_fn, batched=True, remove_columns=['text'])\n",
    "ds_split = ds_tokenized.train_test_split(test_size=0.05, seed=42)\n",
    "train_ds = ds_split['train']\n",
    "eval_ds = ds_split['test']\n",
    "print(f'Train: {len(train_ds)}, Eval: {len(eval_ds)}')\n",
]))

# --- Cell 8: Train ---
cells.append(cell_code([
    "from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling\n",
    "\n",
    "training_args = TrainingArguments(\n",
    "    output_dir=MODEL_OUT,\n",
    "    per_device_train_batch_size=8,\n",
    "    gradient_accumulation_steps=2,\n",
    "    optim='paged_adamw_8bit',\n",
    "    save_steps=200,\n",
    "    save_total_limit=1,\n",
    "    logging_steps=20,\n",
    "    learning_rate=2e-4,\n",
    "    max_grad_norm=0.3,\n",
    "    num_train_epochs=1,\n",
    "    warmup_steps=10,\n",
    "    lr_scheduler_type='cosine',\n",
    "    fp16=True,\n",
    "    eval_strategy='steps',\n",
    "    eval_steps=200,\n",
    ")\n",
    "\n",
    "data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)\n",
    "\n",
    "trainer = Trainer(\n",
    "    model=model,\n",
    "    args=training_args,\n",
    "    train_dataset=train_ds,\n",
    "    eval_dataset=eval_ds,\n",
    "    data_collator=data_collator,\n",
    ")\n",
    "\n",
    "print('Starting training on', torch.cuda.device_count(), 'GPU(s)...')\n",
    "trainer.train()\n",
    "trainer.model.save_pretrained(MODEL_OUT)\n",
    "tokenizer.save_pretrained(MODEL_OUT)\n",
]))

# --- Cell 8: Free GPU memory ---
cells.append(cell_code([
    "print('Training Complete! Freeing memory for model merge...')\n",
    "del model\n",
    "del trainer\n",
    "torch.cuda.empty_cache()\n",
    "gc.collect()\n",
]))

# --- Cell 9: Merge adapter into base model ---
cells.append(cell_code([
    "from transformers import AutoModelForCausalLM\n",
    "from peft import PeftModel\n",
    "\n",
    "print('Loading base model in fp16 on CPU...')\n",
    "base_model = AutoModelForCausalLM.from_pretrained(\n",
    "    model_id,\n",
    "    torch_dtype=torch.float16,\n",
    "    device_map='cpu'\n",
    ")\n",
    "\n",
    "print('Merging LoRA weights...')\n",
    "merged_model = PeftModel.from_pretrained(base_model, MODEL_OUT)\n",
    "merged_model = merged_model.merge_and_unload()\n",
    "\n",
    "merged_model.save_pretrained(MERGED_DIR)\n",
    "tokenizer.save_pretrained(MERGED_DIR)\n",
    "print(f'Merged model saved to {MERGED_DIR}')\n",
    "\n",
    "del base_model\n",
    "del merged_model\n",
    "gc.collect()\n",
]))

# --- Cell 10: Clone & build llama.cpp ---
cells.append(cell_code([
    "%%bash\n",
    "echo 'Cloning llama.cpp...'\n",
    "git clone https://github.com/ggerganov/llama.cpp.git\n",
    "cd llama.cpp\n",
    "make -j4\n",
]))

# --- Cell 11: Convert & quantize ---
cells.append(cell_code([
    "%%bash\n",
    "echo 'Installing llama.cpp python dependencies...'\n",
    "pip install -q -r llama.cpp/requirements.txt\n",
    "\n",
    "echo 'Converting merged model to GGUF (fp16)...'\n",
    "python llama.cpp/convert_hf_to_gguf.py /kaggle/working/gemma_merged_fp16 --outfile /kaggle/working/finetuned-gemma-fp16.gguf --outtype f16\n",
    "\n",
    "echo 'Deleting merged safetensors to free disk space...'\n",
    "rm -rf /kaggle/working/gemma_merged_fp16\n",
    "\n",
    "echo 'Quantizing to Q4_K_M...'\n",
    "./llama.cpp/llama-quantize /kaggle/working/finetuned-gemma-fp16.gguf /kaggle/working/finetuned-gemma-Q4_K_M.gguf Q4_K_M\n",
    "\n",
    "echo 'Deleting fp16 GGUF to free disk space...'\n",
    "rm -f /kaggle/working/finetuned-gemma-fp16.gguf\n",
    "\n",
    "echo 'Done! Final file:'\n",
    "ls -lh /kaggle/working/finetuned-gemma-Q4_K_M.gguf\n",
]))

# --- Cell 12: Upload to HF ---
cells.append(cell_code([
    "from huggingface_hub import HfApi\n",
    "\n",
    "api = HfApi()\n",
    "REPO_ID = 'khedim/NLP-MINI-PROJECT'\n",
    "GGUF_FILE = '/kaggle/working/finetuned-gemma-Q4_K_M.gguf'\n",
    "\n",
    "print(f'Uploading {GGUF_FILE} to {REPO_ID}...')\n",
    "api.upload_file(\n",
    "    path_or_fileobj=GGUF_FILE,\n",
    "    path_in_repo='finetuned-gemma-4-e4b-it-Q4_K_M.gguf',\n",
    "    repo_id=REPO_ID,\n",
    "    commit_message='Upload fine-tuned Gemma 4 E4B-IT Q4_K_M GGUF'\n",
    ")\n",
    "print('Upload Complete! Download this file locally and drop it into models/finetuned-gemma-gguf/')\n",
]))

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py", "mimetype": "text/x-python",
            "name": "python", "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3", "version": "3.12.12"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

with open("notebooks/kaggle_gemma_finetune.ipynb", "w") as f:
    json.dump(notebook, f, indent=2)

print("Notebook generated successfully!")
