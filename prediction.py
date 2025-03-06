from datasets import load_dataset, DatasetDict
import torch
from transformers import DataCollatorForSeq2Seq
import numpy as np
import evaluate
import datetime
from transformers import AutoTokenizer
from transformers import AutoModelForSeq2SeqLM
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer

dataset = load_dataset("abisee/cnn_dailymail","3.0.0")
train_set = dataset["train"].select(range(1000))
val_set = dataset["validation"].select(range(100))
test_set = dataset["test"].select(range(100))
dataset = DatasetDict({
    "train": train_set,
    "validation": val_set,
    "test": test_set
})

model_path = "google-t5/t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_path)


def tokenize_function(examples):
    # breakpoint()
    tokenized_outputs = tokenizer(examples["article"], padding="max_length", truncation=True, max_length=1200)
    outputs = tokenizer(examples["highlights"], padding="max_length", truncation=True)
    # breakpoint()
    return { "input_ids": tokenized_outputs["input_ids"], "attention_mask": tokenized_outputs["attention_mask"], "labels": outputs["input_ids"] }
    
    


tokenized_datasets = dataset.map(tokenize_function, batched=True)
    


save_path = "checkpoint-750"
# Prediction
model = AutoModelForSeq2SeqLM.from_pretrained(save_path)
training_args = Seq2SeqTrainingArguments(
    output_dir="./results",
    per_device_eval_batch_size=4,  # adjust based on your hardware
    predict_with_generate=True,
)
test_trainer = Seq2SeqTrainer(
    model=model, 
    args=training_args,
    data_collator=DataCollatorForSeq2Seq(tokenizer)
)
predictions_output = test_trainer.predict(tokenized_datasets["test"])
summaries = tokenizer.batch_decode(
    predictions_output.predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True
)
print(len(summaries))
print(len(test_set["highlights"]))

with open("summaries_small.txt.src", "w") as f:
    for content in test_set["article"]:
        content = content.replace("\n"," ")
        f.write(content + "\n")

with open("summaries_small_gold.txt.tgt", "w") as f:
    i=1
    for content in test_set["highlights"]:
        content = content.replace("\n"," ")
        print(i)
        i+=1
        f.write(content + "\n")

with open("summaries_small_pred.txt.tgt", "w", encoding="utf-8") as f:
    for summary in summaries:
        f.write(summary + "\n")  # separate summaries with a blank line

print("Summaries saved to summaries.txt")
