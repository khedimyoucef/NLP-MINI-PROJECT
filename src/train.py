from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset
import transformers


def main():
    model_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name)

    ds = load_dataset("json", data_files="data/processed/prefix_stories.json", split="train")

    def tokenize(ex):
        return tokenizer(ex["text"], truncation=True, max_length=256)

    ds = ds.map(tokenize, batched=True, remove_columns=["text"]) 

    args = TrainingArguments(
        output_dir="runs/gpt2-stories",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        fp16=True,
        num_train_epochs=2,
        logging_steps=200,
        save_total_limit=2,
        learning_rate=2e-5,
    )

    trainer = Trainer(model=model, args=args, train_dataset=ds)
    trainer.train()


if __name__ == "__main__":
    main()
