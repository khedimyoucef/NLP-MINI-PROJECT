# 📚 Children's Story Generator

A robust end-to-end pipeline and web application for generating age-conditioned children's stories using fine-tuned Large Language Models (LLMs). This project explores domain adaptation by fine-tuning models like Google's Gemma 4 (E2B-IT) and GPT-2 Small using QLoRA and full fine-tuning techniques, followed by edge deployment via GGUF quantization.

## 🚀 Features
- **Age-Conditioned Generation**: Generates stories tailored to specific age groups (e.g., Age 3-4, Age 11-12) with adapted vocabulary and sentence complexity.
- **Multi-Model Support**: Evaluate and compare between fine-tuned models (`story-gpt2`, `gemma-4-E2B-it-finetuned`) and zero-shot generalists (`gpt2-xl`, `gemma-4-E4B-it-gguf`).
- **Edge Deployment**: Memory-efficient execution utilizing 4-bit quantization (GGUF via `llama.cpp`) to run comfortably on consumer CPU/RAM.
- **Modern Web UI**: A clean, responsive React-based (HTML/JS/CSS) interface powered by FastAPI for interactive story generation.

## 📁 Repository Structure
- `data/` — Raw and processed dataset files.
- `notebooks/` — Jupyter notebooks containing experiments, EDA, and Kaggle training pipelines.
- `src/` — Source code for data preparation, training loops, and evaluation metrics.
- `api/` — FastAPI microservice for model inference and text generation.
- `webapp/` — Minimalistic, modern frontend to demo generation.
- `docker/` — Docker configurations for the API and Web App.
- `scripts/` — Helper bash scripts for model downloading and server management.
- `models/` — Directory for storing local model checkpoints and GGUF files.
- `NLP_Project_Report/` — The complete LaTeX documentation outlining the project's methodology and results.

## 🛠️ Quick Start & Local Reproduction

To fully reproduce this project locally (e.g., on a personal machine without heavy GPU requirements), follow these steps.

### 1. Environment Setup
It is recommended to use an environment manager like `micromamba` or `conda`.
```bash
micromamba create -n NLP python=3.10
micromamba activate NLP
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

### 2. Download Models (Optional)
Downloading all models requires significant storage space. The application is designed to start gracefully even if models are missing, making them optional to download.

**Download Fine-Tuned GPT-2 Small (Recommended for fast CPU inference):**
```bash
export REPO_ID="khedim/NLP-MINI-PROJECT"
export MODEL_DIR="models/story-gpt2"
bash scripts/download_model.sh
```

**Download Fine-Tuned Gemma (GGUF):**
```bash
export REPO_ID="khedim/NLP-MINI-PROJECT"
export MODEL_DIR="models/gemma-4-E2B-it-finetuned"
bash scripts/download_model.sh
```

### 3. Start the API and Web UI
Launch the local FastAPI server using the provided script:
```bash
bash scripts/run_api.sh
```

Once running, navigate to [http://localhost:8000/](http://localhost:8000/) in your web browser. The application will automatically detect the models present in the `models/` directory and list them in the UI.

## 📊 Training Environment
The fine-tuning phase (QLoRA) for the Gemma model was conducted on Kaggle utilizing dual NVIDIA T4 GPUs. You can find the Kaggle-ready benchmark notebook under `notebooks/kaggle_runtime_benchmark.ipynb`.

## 🤝 Contributors
- **Benhammadi Lokmane**
- **Khedim Youcef**