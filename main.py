from datasets import load_dataset, DatasetDict
import torch
from transformers import DataCollatorForSeq2Seq
import numpy as np
import datetime
from transformers import AutoTokenizer
from transformers import AutoModelForSeq2SeqLM
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer


dataset = load_dataset("abisee/cnn_dailymail","3.0.0")
# train_set = dataset["train"].select(range(1000))
# val_set = dataset["validation"].select(range(100))
# test_set = dataset["test"].select(range(100))
# dataset = DatasetDict({
#     "train": train_set,
#     "validation": val_set,
#     "test": test_set
# })

model_path = "google-t5/t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_path)


def tokenize_function(examples):
    # breakpoint()
    tokenized_outputs = tokenizer(examples["article"], padding="max_length", truncation=True, max_length=1000)
    outputs = tokenizer(examples["highlights"], padding="max_length", truncation=True)
    # breakpoint()
    return { "input_ids": tokenized_outputs["input_ids"], "attention_mask": tokenized_outputs["attention_mask"], "labels": outputs["input_ids"] }
    
    


tokenized_datasets = dataset.map(tokenize_function, batched=True)

small_train_dataset = tokenized_datasets["train"].shuffle(seed=42)
small_eval_dataset = tokenized_datasets["validation"].shuffle(seed=42)


model = AutoModelForSeq2SeqLM.from_pretrained(
    model_path, torch_dtype=torch.bfloat16, 
    # attn_implementation='flash_attention_1'
)

date = datetime.datetime.now().strftime("%Y-%m-%d--%H:%M")
save_path = f"output/checkpoint_{date}"
training_args = Seq2SeqTrainingArguments(
    output_dir=save_path,
    eval_strategy="epoch",
    num_train_epochs=3,
    per_device_train_batch_size=3,
    bf16=True,
    predict_with_generate=True,
    # gradient_checkpointing=True
    # deepspeed='ds_config.json'
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=small_train_dataset,
    eval_dataset=small_eval_dataset,
    data_collator= DataCollatorForSeq2Seq(tokenizer),
    # compute_metrics=compute_metrics,
)
trainer.train()
date = datetime.datetime.now().strftime("%Y-%m-%d--%H:%M")