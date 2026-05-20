# 🚀 NLP Project Defense: Live Demo & Model Comparison Guide

This guide contains curated, highly structured evaluation prompts designed to showcase the power of your **children's story generation system**. These prompts are optimized to demonstrate **age-group vocabulary adaptation**, **semantic coherence**, and the differences between **GPT-2 Small (124M)**, **GPT-2 XL (1.5B)**, and **Gemma (2.3B)**.

---

## 📋 Quick-Copy Demo Prompts Cheat Sheet

Use these exact prompts during your live demo to show the professor how the models scale.

| Theme Category | Age Group | Curated Prompt Text (Copy-Paste) | Best For Demonstrating |
| :--- | :--- | :--- | :--- |
| **🐾 Animal Adventures** | **Age 3-4** | `a happy little puppy who lost his bright red ball` | Simple nouns, repetitive sentence structures, high readability. |
| | **Age 7-8** | `a wise old owl teaching a young bird how to fly during a storm` | Compound sentences, moral storytelling. |
| | **Age 11-12** | `a clever squirrel organizing a forest alliance to save their oak tree from winter floods` | Complex narrative, advanced vocab, multiple characters. |
| **🚀 Sci-Fi & Magic** | **Age 3-4** | `a small shiny rocket ship flying to the yellow moon` | High contrast sensory words (shiny, yellow), short actions. |
| | **Age 7-8** | `a curious robot who wants to find out what sweet flowers smell like` | Abstract concepts, sensory explanations, emotional range. |
| | **Age 11-12** | `an astronaut kid landing on Mars who discovers a hidden underground crystal forest` | Complex spatial descriptions, rich vocabulary, high imagination. |
| **❤️ Moral & Social** | **Age 3-4** | `a friendly kid learning how to share toys with friends` | Simple behavioral reinforcement, highly familiar contexts. |
| | **Age 7-8** | `a shy girl who finds her voice to sing in the school talent show` | Emotional maturity, simple internal thoughts, triumph arc. |
| | **Age 11-12** | `a kid who stands up against a classroom bully by organizing a kindness campaign` | Social nuances, advanced resolution, societal/school context. |

---

## 🔬 Model Comparison Guidelines: What to show the Professor

During your 10-minute presentation and 5-minute Q&A, use these specific comparative points to show that you thoroughly understand the technology stack:

### 1. GPT-2 Small (fine-tuned, 124M parameters)
*   **The Focus:** The "efficient baseline."
*   **What to Look For:** 
    *   Generates grammatically correct sentences and adheres to basic storytelling constraints.
    *   *Limitation:* Often repeats certain ideas or gets stuck in localized narrative loops (e.g. repeating the same interaction twice).
    *   *Limitation:* Does not distinguish strongly between Age 3-4 and Age 11-12 (it generally uses a flat, simple vocabulary across all runs due to its small size and vocabulary compression).
    *   *Demo Tip:* Run the **Age 3-4 Puppy** prompt. It will generate a highly readable, cute, but basic story.

### 2. GPT-2 XL (base model, 1.5B parameters)
*   **The Focus:** The "unaligned scale benchmark."
*   **What to Look For:**
    *   A massive leap in grammar fluidity and narrative variance compared to GPT-2 Small.
    *   *Limitation:* Because it is a base, un-finetuned model, it sometimes forgets it is supposed to write a children's story and drifts into writing a book review, a blog post, or a general educational essay.
    *   *Demo Tip:* Show how it struggles to stay purely child-friendly compared to the fine-tuned and guardrailed models.

### 3. Gemma (fine-tuned/quantized GGUF, 2.3B parameters)
*   **The Focus:** The "state-of-the-art production pipeline."
*   **What to Look For:**
    *   **Strict Age Conditioning:** Compare **Age 3-4** (uses basic words like *go, see, run, sad, happy*) vs. **Age 11-12** (uses advanced words like *alliance, subterranean, resonance, anxiety*) for the exact same prompt theme!
    *   **Global Narrative Planning:** The story has a clear introduction, rising action, climax, and a satisfying moral resolution that concludes completely within the max token limit.
    *   **Robust Guardrails:** The instruction-tuned system prompt prevents the model from generating unsafe or non-story content, even if prompted with trick inputs.

---

## 🎯 Steps for an Impressive 5-Minute Live Demo

1.  **Step 1 (The SOTA Showcase):**
    *   Select **Gemma**.
    *   Input: `a curious robot who wants to find out what sweet flowers smell like`
    *   Run once under **Age 3-4**. Show the simple, repetitive sentence structures.
    *   Run again under **Age 11-12**. Point out the complex vocabulary (*olfactory, synthetic, sensory, botanical*) and deep metaphors. This perfectly proves your **Flesch-Kincaid age-conditioning training**!
2.  **Step 2 (Baseline Comparison):**
    *   Select **story-gpt2**.
    *   Input: `a happy little puppy who lost his bright red ball`
    *   Run under **Age 7-8**. Show that it generates a sweet story starting with `"Once upon a time..."` and highlight that it runs beautifully and quickly on consumer CPUs.
3.  **Step 3 (The UI Design flex):**
    *   Toggle the **Dark Mode** switch at the top right to show the premium, responsive interface design that adapts to the system's preferences.
