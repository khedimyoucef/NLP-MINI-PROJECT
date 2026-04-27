# Children's Story Generator (mini-project)

Project scaffold for fine-tuning GPT-2 small to generate age-conditioned children's stories.

Structure:
- `data/` — raw and processed dataset files
- `notebooks/` — experiments and EDA
- `src/` — data prep, training and evaluation code
- `api/` — FastAPI microservice for generation
- `webapp/` — minimal frontend to demo generation
- `docker/` — Dockerfile(s) for API and webapp
- `scripts/` — helper run scripts

See `src/` and `api/` for starter scripts. Use `requirements.txt` to install deps.

Runtime estimate:
- Run `python -m src.benchmark_runtime` to estimate 1xT4 vs 2xT4 training time.
- Pass `--single-gpu-tokens-per-sec` if you already measured throughput on one GPU.

Kaggle notebook:
- Upload [notebooks/kaggle_runtime_benchmark.ipynb](notebooks/kaggle_runtime_benchmark.ipynb) to Kaggle and run the cells top to bottom.

Local CPU run:
- Install dependencies in your micromamba env (CPU-only torch recommended):

```bash
micromamba run -n NLP python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
micromamba run -n NLP python -m pip install -r requirements.txt
```

- Download the fine-tuned model once:

```bash
PYTHON_BIN=/home/youcef/micromamba/envs/NLP/bin/python bash scripts/download_model.sh
```

- Start the API + UI:

```bash
PYTHON_BIN=/home/youcef/micromamba/envs/NLP/bin/python bash scripts/run_api.sh
```

Then open http://localhost:8000/ to use the web UI.