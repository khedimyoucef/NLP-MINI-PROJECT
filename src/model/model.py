from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model(model_name="gpt2"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return model, tokenizer
