import json

notebook = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Kaggle End-to-End: Fine-Tune & Quantize Gemma 4 E4B-IT\n",
                "\n",
                "This notebook performs the complete pipeline:\n",
                "1. Downloads the Children Stories dataset.\n",
                "2. Fine-tunes `google/gemma-4-E4B-it` using QLoRA on 2x T4 GPUs.\n",
                "3. Merges the LoRA adapter back into the base model.\n",
                "4. Uses `llama.cpp` to convert the merged model into an fp16 `.gguf` file.\n",
                "5. Quantizes the file into a `Q4_K_M` format (approx ~5GB).\n",
                "6. Uploads **ONLY** the final `.gguf` file to your Hugging Face repo."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "!pip install -q -U git+https://github.com/huggingface/transformers.git peft bitsandbytes trl accelerate datasets huggingface_hub"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import os\n",
                "import sys\n",
                "import subprocess\n",
                "import torch\n",
                "import gc\n",
                "\n",
                "BASE_DIR = '/kaggle/working'\n",
                "MODEL_OUT = os.path.join(BASE_DIR, 'gemma_lora_output')\n",
                "MERGED_DIR = os.path.join(BASE_DIR, 'gemma_merged_fp16')\n",
                "os.makedirs(MODEL_OUT, exist_ok=True)\n",
                "os.makedirs(MERGED_DIR, exist_ok=True)"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "from huggingface_hub import login\n",
                "from kaggle_secrets import UserSecretsClient\n",
                "import wandb\n",
                "\n",
                "os.environ['WANDB_DISABLED'] = 'true'\n",
                "\n",
                "try:\n",
                "    user_secrets = UserSecretsClient()\n",
                "    hf_token = user_secrets.get_secret('HF_TOKEN')\n",
                "    login(token=hf_token)\n",
                "    print('Logged in with HF_TOKEN from Kaggle Secrets')\n",
                "except Exception as e:\n",
                "    if 'HF_TOKEN' in os.environ:\n",
                "        login(token=os.environ['HF_TOKEN'])\n",
                "    else:\n",
                "        print('WARNING: No HF_TOKEN found! You must login to push to Hugging Face!')\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "from datasets import load_dataset\n",
                "\n",
                "ds = load_dataset(\"ajibawa-2023/Children-Stories-Collection\", split='train', cache_dir=os.path.join(BASE_DIR, 'hf_cache'))\n",
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
                "    text = f\"<start_of_turn>user\\n{prefix}\\n{prompt}<end_of_turn>\\n<start_of_turn>model\\n{story}<end_of_turn>\\n\"\n",
                "    return {'text': text}\n",
                "\n",
                "ds_formatted = ds_sample.map(format_gemma_instruct, with_indices=True, remove_columns=ds_sample.column_names)\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig\n",
                "\n",
                "model_id = \"google/gemma-4-E4B-it\"\n",
                "\n",
                "bnb_config = BitsAndBytesConfig(\n",
                "    load_in_4bit=True,\n",
                "    bnb_4bit_use_double_quant=True,\n",
                "    bnb_4bit_quant_type=\"nf4\",\n",
                "    bnb_4bit_compute_dtype=torch.float16\n",
                ")\n",
                "\n",
                "tokenizer = AutoTokenizer.from_pretrained(model_id)\n",
                "model = AutoModelForCausalLM.from_pretrained(\n",
                "    model_id,\n",
                "    quantization_config=bnb_config,\n",
                "    device_map=\"balanced\",\n",
                "    max_memory={0: \"12GB\", 1: \"12GB\"}\n",
                ")"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "from peft import LoraConfig, get_peft_model\n",
                "\n",
                "model.gradient_checkpointing_enable()\n",
                "model.enable_input_require_grads()\n",
                "\n",
                "# Manually prepare for kbit training without upcasting massive embeddings\n",
                "for name, param in model.named_parameters():\n",
                "    param.requires_grad = False\n",
                "    # Only cast layer norms to fp32 for stability, leave embeddings in fp16\n",
                "    if param.ndim == 1 and \"norm\" in name.lower():\n",
                "        param.data = param.data.to(torch.float32)\n",
                "\n",
                "lora_config = LoraConfig(\n",
                "    r=16,\n",
                "    lora_alpha=32,\n",
                "    target_modules=[\"q_proj\", \"k_proj\", \"v_proj\", \"o_proj\", \"gate_proj\", \"up_proj\", \"down_proj\"],\n",
                "    lora_dropout=0.05,\n",
                "    bias=\"none\",\n",
                "    task_type=\"CAUSAL_LM\"\n",
                ")\n",
                "\n",
                "model = get_peft_model(model, lora_config)\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "from trl import SFTTrainer\n",
                "from transformers import TrainingArguments\n",
                "\n",
                "training_args = TrainingArguments(\n",
                "    output_dir=MODEL_OUT,\n",
                "    per_device_train_batch_size=4,\n",
                "    gradient_accumulation_steps=4,\n",
                "    optim=\"paged_adamw_8bit\",\n",
                "    save_steps=100,\n",
                "    logging_steps=20,\n",
                "    learning_rate=2e-4,\n",
                "    max_grad_norm=0.3,\n",
                "    num_train_epochs=1,\n",
                "    warmup_ratio=0.03,\n",
                "    lr_scheduler_type=\"cosine\",\n",
                "    fp16=True,\n",
                "    ddp_find_unused_parameters=False\n",
                ")\n",
                "\n",
                "trainer = SFTTrainer(\n",
                "    model=model,\n",
                "    train_dataset=ds_formatted,\n",
                "    peft_config=lora_config,\n",
                "    dataset_text_field=\"text\",\n",
                "    max_seq_length=512,\n",
                "    tokenizer=tokenizer,\n",
                "    args=training_args,\n",
                ")\n",
                "\n",
                "trainer.train()\n",
                "trainer.model.save_pretrained(MODEL_OUT)\n",
                "tokenizer.save_pretrained(MODEL_OUT)"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "print(\"Training Complete! Freeing memory for model merge...\")\n",
                "del model\n",
                "del trainer\n",
                "torch.cuda.empty_cache()\n",
                "gc.collect()"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "from transformers import AutoModelForCausalLM\n",
                "from peft import PeftModel\n",
                "\n",
                "print(\"Loading base model in fp16 on CPU...\")\n",
                "base_model = AutoModelForCausalLM.from_pretrained(\n",
                "    model_id,\n",
                "    torch_dtype=torch.float16,\n",
                "    device_map=\"cpu\" # Use Kaggle CPU RAM (30GB) to avoid OOM\n",
                ")\n",
                "\n",
                "print(\"Merging LoRA weights...\")\n",
                "merged_model = PeftModel.from_pretrained(base_model, MODEL_OUT)\n",
                "merged_model = merged_model.merge_and_unload()\n",
                "\n",
                "merged_model.save_pretrained(MERGED_DIR)\n",
                "tokenizer.save_pretrained(MERGED_DIR)\n",
                "print(f\"Merged model saved to {MERGED_DIR}\")\n",
                "\n",
                "del base_model\n",
                "del merged_model\n",
                "gc.collect()"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "%%bash\n",
                "echo \"Cloning llama.cpp...\"\n",
                "git clone https://github.com/ggerganov/llama.cpp.git\n",
                "cd llama.cpp\n",
                "make -j4"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "%%bash\n",
                "echo \"Installing llama.cpp python dependencies...\"\n",
                "pip install -q -r llama.cpp/requirements.txt\n",
                "\n",
                "echo \"Converting merged model to GGUF (fp16)...\"\n",
                "python llama.cpp/convert_hf_to_gguf.py /kaggle/working/gemma_merged_fp16 --outfile /kaggle/working/finetuned-gemma-fp16.gguf --outtype f16\n",
                "\n",
                "echo \"Deleting merged safetensors to free up 9GB of disk space before quantizing...\"\n",
                "rm -rf /kaggle/working/gemma_merged_fp16\n",
                "\n",
                "echo \"Quantizing to Q4_K_M...\"\n",
                "./llama.cpp/llama-quantize /kaggle/working/finetuned-gemma-fp16.gguf /kaggle/working/finetuned-gemma-Q4_K_M.gguf Q4_K_M\n",
                "\n",
                "echo \"Deleting huge fp16 GGUF to free up another 9GB of disk space...\"\n",
                "rm -f /kaggle/working/finetuned-gemma-fp16.gguf"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "from huggingface_hub import HfApi\n",
                "\n",
                "api = HfApi()\n",
                "REPO_ID = 'khedim/NLP-MINI-PROJECT'\n",
                "GGUF_FILE = '/kaggle/working/finetuned-gemma-Q4_K_M.gguf'\n",
                "\n",
                "print(f\"Uploading {GGUF_FILE} to {REPO_ID}...\")\n",
                "api.upload_file(\n",
                "    path_or_fileobj=GGUF_FILE,\n",
                "    path_in_repo=\"finetuned-gemma-4-e4b-it-Q4_K_M.gguf\",\n",
                "    repo_id=REPO_ID,\n",
                "    commit_message=\"Upload locally fine-tuned Gemma 4 E4B-IT Q4_K_M GGUF model\"\n",
                ")\n",
                "print(\"Upload Complete! You can now download this single file locally and drop it into your UI!\")"
            ]
        }
    ],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {
                "name": "ipython",
                "version": 3
            },
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.12.12"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

with open("notebooks/kaggle_gemma_finetune.ipynb", "w") as f:
    json.dump(notebook, f, indent=2)

print("Notebook generated successfully!")
