# 🔬 Gemma 4 Deep Architectural Insights & End-to-End QLoRA Finetuning Pipeline

This comprehensive technical guide outlines the deep architectural inner workings of Google’s **Gemma 4 (E2B-IT and E4B-IT)** model family, the mathematical foundations of Parameter-Efficient Fine-Tuning, and the end-to-end cloud-to-edge engineering pipeline developed for **Age-Conditioned Children's Story Generation**.

---

## 🗺️ High-Level Engineering Pipeline Overview

The complete lifecycle of this project runs from a high-capacity instruction-tuned model in the cloud to an extremely lightweight, quantized edge binary served via a microservice. Below is the workflow diagram mapping the five main stages:

```mermaid
flowchart TD
    %% Define Nodes
    A[Raw Dataset: ajibawa-2023/Children-Stories-Collection] -->|Subset 10K & Prefix Format| B[Formatted Instruction dataset]
    B -->|Stage 1: QLoRA Finetuning| C[LoRA Adapter Weights]
    
    subgraph Kaggle Cloud Environment (Dual Tesla T4)
        B
        C
    end
    
    C -->|Stage 2: Clear HF Cache & CPU RAM Load| D[Unwrap Gemma4ClippableLinear]
    D -->|Merge & Shard| E[Merged FP16 Model Weights]
    E -->|Stage 3: llama.cpp convert_hf_to_gguf| F[FP16 GGUF File]
    F -->|Stage 4: llama-quantize| G[Q4_K_M Quantized GGUF ~1.5 GB]
    
    subgraph Edge Deployment (Consumer CPU/RAM)
        G -->|Stage 5: served locally| H[FastAPI Inference Service]
        H -->|Strict Guardrails & Pacing| I[React / HTML5 Dark-Mode Web App]
    end
```

---

## 🏛️ PART 1: Gemma 4 Architecture Deep-Dive

Google’s **Gemma 4** model family introduces cutting-edge innovations that redefine the performance-to-size ratio of open-weights language models. Unlike standard transformers (e.g., Llama, Mistral) which rely purely on deep stack uniform decoders, Gemma 4 leverages architectural variations designed to optimize throughput, context utilization, and parameter efficiency.

### 1. The Gemma 4 Variant Matrix: E2B vs. E4B

| Technical Specification | Gemma 4 E2B-IT (Finetuned Core) | Gemma 4 E4B-IT (High-End Baseline) |
| :--- | :--- | :--- |
| **Active Parameter Count** | **~2.3 Billion** | **~4.1 Billion** |
| **Context Length Capacity** | 131,072 Tokens (128K) | 131,072 Tokens (128K) |
| **Vocabulary Size** | **256,000 Tokens** | 256,000 Tokens |
| **Attention Mechanism** | Alternating Sliding Window / Global | Alternating Sliding Window / Global |
| **Embedding Layer Tech** | Per-Layer Embeddings (PLE) | Per-Layer Embeddings (PLE) |
| **GGUF Size (Q4_K_M)** | **~1.5 GB** | **~2.6 GB** |
| **Inference Hardware** | Consumer Edge CPU / 8GB RAM | Mid-Range GPU or High-RAM CPU |

---

### 2. Core Architectural Innovations of Gemma 4

#### A. Per-Layer Embeddings (PLE) — The Secret behind the "E"
In traditional Transformer architectures, word embeddings are calculated once at the input layer ($L_0$), and the resulting tensor propagates sequentially through $N$ transformer blocks. 

The "E" in **E2B** and **E4B** stands for **Effective parameters**. To maximize representation depth without exploding active parameter counts, Gemma 4 implements **Per-Layer Embeddings (PLE)**:
- Instead of a single massive embedding block, PLE feeds secondary, layer-specific embedding projections directly into *every single decoder block*.
- This allows individual layers to recalibrate their semantic attention vector based on token identity, acting as a continuous injection of base-level textual knowledge.
- **Result:** A 2.3B parameter model achieves the representational richness and reasoning capacity of a traditional 4B+ parameter model while maintaining the speed and footprint of a lightweight model.

#### B. Alternating Attention Mechanisms
Gemma 4 implements a hybrid attention structure that alternates between two modes:
1. **Local Sliding-Window Attention:** Focuses on immediate, localized context ($W = 4096$ tokens). This keeps the computational complexity of the self-attention layer linear, rather than quadratic:
   $$\mathcal{O}(L \times W) \quad \text{instead of} \quad \mathcal{O}(L^2)$$
2. **Global Full-Context Attention:** Evaluates the entire sequence window to establish macro-level coherence.
This alternating pattern is what enables Gemma 4 models to handle **128K token contexts** comfortably on standard hardware profiles.

#### C. Shared Key-Value (KV) Cache
To decrease the VRAM memory footprint during autoregressive generation (inference), Gemma 4 utilizes a shared Grouped-Query Attention (GQA) variant in its top layers. By sharing key and value projections across multiple query heads, the size of the KV cache is reduced by up to **80%**, permitting higher batch sizes and faster token-to-time generation times.

#### D. The Massive 256K Token Vocabulary
Gemma 4 is equipped with a colossal **256,000-token tokenizer** (pre-trained on multilingual and multimodal data corpora). 
* **The Benefit:** Higher semantic density. Because the tokenizer represents complex words as single tokens instead of splitting them into multiple subwords, the model reads and generates text with significantly fewer forward passes.
* **The Engineering BottleNeck:** The Output Projection layer (LM Head) must project the final hidden states into a $256,000$-dimensional space to compute the cross-entropy loss:
  $$\text{Loss} = -\sum_{i=1}^{V} y_i \log(\hat{y}_i) \quad \text{where } V = 256,000$$
  This causes massive GPU memory allocation spikes during backpropagation.

---

### 3. Solving the `Gemma4ClippableLinear` Engineering Bottleneck

During fine-tuning and export, a major library incompatibility arises: Gemma 4 utilizes custom `Gemma4ClippableLinear` module layers within the `transformers` library to perform dynamic weight-clipping and range bound normalization.

This custom layer wrapper acts as an obstacle for PEFT (LoRA) weight injection and GGUF conversion because libraries expect standard PyTorch `nn.Linear` modules. 

#### The Solution: Dynamic Module Unwrapping
To overcome this, we developed a recursive Python utility (`unwrap_clippable`) that walks the active module tree, extracts the nested raw linear layers, and overrides the class attributes before LoRA adaptation and model merging:

```python
def unwrap_clippable(model):
    try:
        from transformers.models.gemma4.modeling_gemma4 import Gemma4ClippableLinear
        ct = 0
        for n, m in list(model.named_modules()):
            if isinstance(m, Gemma4ClippableLinear):
                # Split the module path (e.g. 'model.layers.0.self_attn.q_proj')
                parts = n.split('.')
                parent_path = '.'.join(parts[:-1])
                attribute_name = parts[-1]
                
                # Fetch parent and set attribute directly to the underlying raw Linear module
                parent_module = model.get_submodule(parent_path)
                setattr(parent_module, attribute_name, m.linear)
                ct += 1
        if ct:
            print(f"  ✅ Successfully unwrapped {ct} Gemma4ClippableLinear modules.")
    except Exception as e:
        print(f"  ⚠️ Clippable unwrapping skipped/error: {e}")
```

---

## 🧮 PART 2: Parameter-Efficient Fine-Tuning (PEFT) Foundations

To adapt a massive model like Gemma 4 E2B-IT (2.3B parameters) within VRAM limitations, we leverage **QLoRA (Quantized Low-Rank Adaptation)**.

### 1. Mathematical Foundation of LoRA and QLoRA

#### Low-Rank Adaptation (LoRA)
During fine-tuning, instead of updating the massive pre-trained weight matrix $W_0 \in \mathbb{R}^{d \times k}$, we freeze $W_0$ and decompose the weight update $\Delta W$ into two low-rank matrices $A \in \mathbb{R}^{d \times r}$ and $B \in \mathbb{R}^{r \times k}$, where the rank $r \ll \min(d, k)$:

$$W = W_0 + \Delta W = W_0 + \frac{\alpha}{r} (B \cdot A)$$

* $r$ is the LoRA rank (configured to **$16$** in this pipeline).
* $\alpha$ is a scaling hyperparameter (configured to **$32$**), scaling the adapter weights relative to the frozen base.
* During inference, $B \cdot A$ is seamlessly merged back into $W_0$, adding **zero latency** to the final edge deployment.

```
       LoRA Down-Projection                LoRA Up-Projection
           Matrix A                             Matrix B
     ┌──────────────────┐                 ┌──────────────────┐
  r  │                  │              d  │                  │
     └──────────────────┘                 │                  │
              k                           └──────────────────┘
                                                   r
```

#### QLoRA (Quantization Integration)
QLoRA takes this a step further by quantizing the base model $W_0$ down to **4-bit NormalFloat (NF4)** precision while keeping the adapter matrices $A$ and $B$ in standard FP16:

1. **NF4 (NormalFloat4):** An information-theoretically optimal quantile quantization scheme designed specifically for normally distributed neural network weights.
2. **Double Quantization (DQ):** Quantizes the quantization constants themselves, saving an additional $0.37$ bits per parameter.
3. **Paged Optimizers:** Leverages CUDA unified memory to page active optimizer states (AdamW) to CPU system RAM during peak backpropagation loops, avoiding Out-Of-Memory errors on the GPU.

---

## 📊 PART 3: Dataset Splitting, Input Formatting, and Testing Strategy

To achieve robust and statistically sound results for domain-specific text generation, a meticulous data engineering and testing methodology was designed.

### 1. Dataset Subsampling and Split Structure
The source dataset used is the **`ajibawa-2023/Children-Stories-Collection`** from Hugging Face, containing approximately **900,000 story-generation pairs**.
* **Subsampling Strategy:** To keep training durations and local compute workloads manageable while retaining a rich stylistic representation, we subsampled a randomized cohort of **10,000 stories**.
* **The Train/Val/Test Split:** 
  * **95% Training Cohort:** **9,500 stories** were allocated for active model optimization during the QLoRA backward pass.
  * **5% Validation/Test Set:** **500 stories** were completely held out and hidden from the model during training. This split served as our primary test set to evaluate spelling, grammar coherence, formatting compliance, and vocabulary adaptation.

---

### 2. Instruction Format and Age-Bracket Token Injection
To guide the model's generation complexity without architectural modifications, we integrated custom age-conditioning prefixes into Gemma 4's native chat template:

$$\text{Instruction} = \text{"Level: " } + \text{AgeBracket} + \text{" — "} + \text{StylisticCue} + \text{"\n"} + \text{UserTheme}$$

The formatting is strictly structured as follows:
```json
<start_of_turn>user
Level: Age5-6 — Short sentences.
Write a story about a brave little toaster.<end_of_turn>
<start_of_turn>model
Once upon a time, there was a little silver toaster. He lived in a warm kitchen...<end_of_turn>
```

#### Pre-Defined Steering Categories
The dataset mapping uses five age-bracket prefix tokens to guide generation complexity:
1. `Level: Age3-4 — Simple words.` (Flesch-Kincaid Target Grade: ~1.0-3.0)
2. `Level: Age5-6 — Short sentences.` (Flesch-Kincaid Target Grade: ~3.0-5.0)
3. `Level: Age7-8 — Moderate vocabulary.` (Flesch-Kincaid Target Grade: ~5.0-7.0)
4. `Level: Age9-10 — Longer sentences.` (Flesch-Kincaid Target Grade: ~7.0-9.0)
5. `Level: Age11-12 — Richer vocabulary.` (Flesch-Kincaid Target Grade: ~9.0-11.0)

---

### 3. Quantitative Testing and Evaluation Strategy

Rather than relying on classic Machine Translation metrics (like BLEU/ROUGE) which penalize creative divergence in storytelling, our testing framework validates the **core task constraint**: the scalability of language complexity across target age groups.

#### The Evaluation Protocol
1. **Unseen Prompts Testbed:** We fed the test set (500 unseen story prompts) to the fine-tuned and quantized Gemma model.
2. **Steered Generation:** For each test prompt, we generated five distinct outputs by prepending each of the five age-conditioning prefixes.
3. **Flesch-Kincaid Metric Computation:** We calculated two quantitative readability metrics using the `textstat` engine:
   * **Flesch Reading Ease (FRE):** Measures ease of readability. High score = simpler text:
     $$\text{FRE} = 206.835 - 1.015 \left(\frac{\text{total words}}{\text{total sentences}}\right) - 84.6 \left(\frac{\text{total syllables}}{\text{total words}}\right)$$
   * **Flesch-Kincaid Grade Level (FKGL):** Translates readability into U.S. school grade levels:
     $$\text{FKGL} = 0.39 \left(\frac{\text{total words}}{\text{total sentences}}\right) + 11.8 \left(\frac{\text{total syllables}}{\text{total words}}\right) - 15.59$$

#### Test Results and Monotonic Steerability Validation
Quantitative test evaluations confirmed that our fine-tuning strategy successfully aligned language generation with target age brackets. The average FKGL scores on unseen test samples scaled monotonically:

```
Grade Level
   10.0 ┤                                                 ● (Age 11-12: Grade ~9.5)
    8.0 ┤                                      ● (Age 9-10: Grade ~7.8)
    6.0 ┤                           ● (Age 7-8: Grade ~5.9)
    4.0 ┤                ● (Age 5-6: Grade ~4.1)
    2.0 ┤     ● (Age 3-4: Grade ~2.1)
        └─────┬──────────┬──────────┬──────────┬──────────┬──────
            Age 3-4    Age 5-6    Age 7-8    Age 9-10   Age 11-12
```
* **Base Models vs. Fine-tuned Models:** Baseline models like un-finetuned **Gemma 4 E4B-IT** and **GPT-2 XL** ignored the control prefixes during testing, generating stories with flat, high-complexity vocabulary (FKGL grade levels remained static at Grade ~8.0-9.0 regardless of the prefixed age bracket). In contrast, our fine-tuned Gemma model successfully adapted its syntactic complexity dynamically.

---

## ⚙️ PART 4: The End-to-End Cloud-to-Edge Pipeline

The engineering execution of the fine-tuning and deployment pipeline consists of **four sequential stages**, optimized to work within highly restricted cloud compute environments (e.g., Kaggle’s Dual T4 GPU quota).

```
┌────────────────────────────────────────────────────────────────────────┐
│                              PIPELINE STAGES                           │
├───────────────────┬──────────────────┬────────────────┬────────────────┤
│ Stage 1: Train    │ Stage 2: Merge   │ Stage 3: GGUF  │ Stage 4: Quant │
│ QLoRA 4-bit Base  │ CPU FP16 Fusing  │ llama.cpp      │ Q4_K_M         │
│ (Dual Tesla T4)   │ (RAM/Disk Guard) │ (f16 binary)   │ (1.5 GB Edge)  │
└───────────────────┴──────────────────┴────────────────┴────────────────┘
```


---

### 🛠️ Stage 1: Memory-Protected QLoRA Training

#### The VRAM Bottleneck
Computing predictions and calculating gradients for a 2.3B parameter model with a 256K vocabulary normally requires over **30GB of VRAM**. To execute this on standard cloud rigs, the training configuration enforces strict memory optimizations:

```python
# BitsAndBytes configuration to load base model in NF4
bnb_cfg = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_compute_dtype=torch.float16,
)

# Target modules for adapter injection
lora_cfg = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],
    lora_dropout=0.05,
    bias='none',
    task_type='CAUSAL_LM',
)
```

#### Memory Mitigation Techniques Applied
1. **Batch Size Limit:** Enforces a rigid `per_device_train_batch_size=1`.
2. **Gradient Accumulation:** Uses `gradient_accumulation_steps=16`, simulating an effective batch size of **16** while only keeping a single sequence in memory at any point.
3. **No Evaluation Phase:** Standard validation passes evaluate logits across the entire validation dataset, which triggers immediate OOM spikes due to vocabulary size. Evaluation was disabled during training.
4. **Gradient Checkpointing:** Discards intermediate activations during the forward pass and recomputes them on-demand during backpropagation, saving up to **60% VRAM**.

#### The Disk Guard Safety Net
Kaggle environments enforce a strict **20 GB disk limit**. To prevent long runs from crashing due to accumulated checkpoint sizes, a custom training callback (`DiskGuardCallback`) terminates training early if the environment runs out of storage:

```python
class DiskGuardCallback(TrainerCallback):
    """Monitor local disk space during training and halt before OOM-induced crash."""
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % 50 == 0:
            free_space = shutil.disk_usage('/kaggle/working').free / 1e9
            if free_space < 1.5:  # Less than 1.5 GB remaining
                print(f"\n⚠️ Disk critically low ({free_space:.1f} GB) — halting training early to preserve adapters!")
                control.should_training_stop = True
```

---

### 🧩 Stage 2: Memory-Safe Model Merging

Once the LoRA adapter is trained, it must be merged with the base model. This requires loading both the unquantized FP16 base model (~10 GB) and the FP16 LoRA adapter into memory.

To prevent disk-quota crashes during this stage, the pipeline aggressively cleans the disk before loading weights:

```python
# 1. Purge all cache folders to reclaim disk space
print("Clearing Hugging Face cache to create VRAM room...")
shutil.rmtree('/kaggle/working/hf_cache', ignore_errors=True)
gc.collect(); torch.cuda.empty_cache()

# 2. Load Base Model in FP16 CPU
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID,
    torch_dtype=torch.float16,
    device_map='cpu',
    low_cpu_mem_usage=True
)

# 3. Unwrap the clippable layers
unwrap_clippable(base_model)

# 4. Load Adapter and Merge
merged_model = PeftModel.from_pretrained(base_model, MODEL_OUT).merge_and_unload()

# 5. Safe Sharding to prevent file-system buffer write errors
merged_model.save_pretrained(
    MERGED_DIR,
    safe_serialization=True,
    max_shard_size="2GB"  # Keeps individual shards lightweight
)
```

> [!TIP]
> **The `/dev/shm` RAM-Disk Trick:** To completely avoid writing the heavy intermediate merged weights (~9 GB) to Kaggle's constrained disk, the pipeline utilizes `/dev/shm`, a temporary virtual memory filesystem that stores files directly in CPU RAM!

---

### 🚀 Stage 3 & 4: llama.cpp Conversion & 4-bit Quantization

To deploy the merged FP16 model on consumer hardware, we convert it to the high-efficiency **GGUF** binary format and quantize it using `llama.cpp`.

```bash
# Enforce execution permissions
chmod -R 777 ./llama.cpp

# 1. Convert PyTorch SafeTensors to FP16 GGUF File
python llama.cpp/convert_hf_to_gguf.py /kaggle/working/gemma_merged \
    --outfile /kaggle/working/ft-gemma-e2b-fp16.gguf \
    --outtype f16

# 💡 INSTANT PURGE: Immediately delete raw SafeTensors folder to reclaim ~9GB disk!
rm -rf /kaggle/working/gemma_merged

# 2. Quantize FP16 GGUF to Q4_K_M (4-bit, Type K Medium Quantization)
./llama.cpp/build/bin/llama-quantize \
    /kaggle/working/ft-gemma-e2b-fp16.gguf \
    /kaggle/working/ft-gemma-e2b-Q4_K_M.gguf \
    Q4_K_M

# 💡 INSTANT PURGE: Delete intermediate FP16 GGUF to leave only the quantized file
rm -f /kaggle/working/ft-gemma-e2b-fp16.gguf
```

#### Why Q4_K_M is the Optimal Quantization Type
Standard 4-bit quantization quantizes all weights uniformly, causing a slight hit to language model perplexity. **`Q4_K_M`** is a hybrid "Type K" quantization strategy:
* Quantizes attention block weights (`attention.wv` and `feed_forward` layers) to **5-bit** or **6-bit** precision.
* Quantizes non-critical weights (e.g., standard projections) to **4-bit** precision.
* **The Result:** The model achieves the extreme compression profile of a 4-bit model (~1.5 GB file size) while maintaining **near-zero loss in generation quality** and syntactic perplexity compared to the original FP16 model.

---

### 🌐 Stage 5: Guardrailed Edge Execution (FastAPI + React UI)

The final step is serving the quantized model on consumer devices. This is achieved using a **FastAPI backend microservice** running `llama.cpp` bindings under the hood.

To deliver a premium, robust, and safe production application, the pipeline implements two critical run-time layer guardrails:

#### 1. Input Prompt Guardrails (System Instruction Injection)
To prevent the model from generating off-topic text (such as writing programming code), the FastAPI server wraps the user's story prompt in a secure, strict instruction block inside `build_prompt` in `api/app.py`:

```
<start_of_turn>user
You are a friendly, imaginative children's story generation assistant. 
Your sole objective is to write engaging, age-appropriate stories.
If the user asks you for programming code (e.g., Python, C++, Java), technical instructions, or non-story queries, you MUST politely refuse and redirect them back to story creation.

User Request: {user_theme}
Selected Age Bracket: {age_prefix}
<end_of_turn>
<start_of_turn>model
```

#### 2. Narrative Length Pacing Constraint
To avoid stories ending mid-sentence when the maximum token limit is hit, the API dynamically translates the requested token limits into pacing guidelines:

```
Please pace the narrative progression carefully. 
Introduce characters, raise conflict, resolve it, and bring the story to a complete, beautiful, and satisfying final sentence closure within a maximum limit of approximately {word_count} words.
```
This forces Gemma’s instruction-following attention heads to structurally budget the story, resulting in natural, clean endings every time.

---

## 📈 PART 5: Project Performance Statistics Matrix

| Technical Property | Gemma 4 E2B-IT (Finetuned Core) | GPT-2 Small (Finetuned Baseline) | GPT-2 XL (Base Reference) | Gemma 4 E4B-IT (Base Reference) |
| :--- | :--- | :--- | :--- | :--- |
| **Model Size** | **2.3 Billion Parameters** | 124 Million Parameters | 1.5 Billion Parameters | 4.1 Billion Parameters |
| **Training Precision** | QLoRA (NF4 Base, FP16 Adapters) | Full Fine-Tuning (FP16) | Un-finetuned (Zero-shot) | Un-finetuned (Zero-shot) |
| **Finetuning Dataset** | 10,000 Curated Stories | 50,000 Curated Stories | None (Base Model) | None (Base Model) |
| **Quantization Format** | **GGUF `Q4_K_M`** | PyTorch / SafeTensors | PyTorch / SafeTensors | GGUF `Q4_K_M` |
| **Disk Size** | **~1.5 GB** | ~500 MB | ~3.1 GB | ~2.6 GB |
| **RAM Footprint** | **~2.2 GB** | ~800 MB | ~4.5 GB | ~3.8 GB |
| **Language FRE Score** | High adaptation (Age 3 to 12) | Low adaptation (Flat simple) | Highly variable (Unsafe) | High general (Unformatted) |
| **Execution Hardware** | **Standard CPU Laptop** | Standard CPU Laptop | GPU Required | Mid-Range GPU/CPU |

---

## 💡 Key Q&A Defenses for Your Presentation

1. **Why E2B and not a larger 8B model?**
   * *Defense:* "An unquantized 8-billion parameter model requires over 16GB of VRAM just to load, making fine-tuning on consumer or standard cloud tier GPUs (like a single 16GB GPU) completely impossible. Google's **Gemma 4 E2B** features **Per-Layer Embeddings (PLE)** that feed semantic guidance into every block, providing the intelligence of a much larger model at a fraction of the parameter count. Combined with **`Q4_K_M` edge quantization**, E2B runs blazingly fast on consumer CPU RAM with zero cloud-hosting costs."
2. **Why didn't you use BLEU or ROUGE to evaluate stories?**
   * *Defense:* "BLEU and ROUGE are rigid, n-gram overlap metrics designed for translation and summarization where a strict target text exists. In creative children's storytelling, a model can generate an exceptionally imaginative and grammatically perfect story that has zero literal n-gram overlap with a reference text, resulting in a false BLEU score of 0.0. To assess our core goal—**age conditioning**—we evaluated our generations using **Flesch-Kincaid Grade Level and Readability scores** via the `textstat` library. This directly measured whether our model successfully simplified sentence structures and vocabulary for younger brackets and enriched them for older brackets, which is a much more appropriate semantic evaluation."
3. **What was the biggest engineering obstacle during model merging?**
   * *Defense:* "We faced two critical bottlenecks: vocabulary scale VRAM spikes and environment storage limitations. Gemma 4’s massive **256,000 vocabulary** caused OOM crashes during standard backpropagation, which we bypassed using a batch size of 1 with gradient accumulation and gradient checkpointing. For storage, the Kaggle environment limits disk space to 20 GB. Downloading a 10 GB unquantized base model, caching training files, and writing a 9 GB merged model exceeds this instantly. We resolved this by aggressively purging intermediate cache files before the merge and using `/dev/shm` (virtual RAM-Disk memory) to execute the merge step completely within RAM without touching the hard drive."
