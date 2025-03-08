import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import GPT2Config, GPT2LMHeadModel, AdamW, get_scheduler
from datasets import load_dataset, DatasetDict
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
from torch.amp import autocast, GradScaler
import numpy as np
from tqdm import tqdm
import os
import json
from tokenizers import processors
from model_custom import DecoderOnlyTransformer, SummarizationDataset, CustomTokenizer

# Main function
def main():
    BATCH_SIZE = 8
    GRADIENT_ACCUMULATION_STEPS = 8
    LEARNING_RATE = 5e-5
    EPOCHS = 1
    MAX_SEQ_LEN = 1000
    HEAD_LEN = 600
    TAIL_LEN = 300
    VOCAB_SIZE = 50257  # Adjust based on your tokenizer
    MODEL_DIM = 768
    NUM_HEADS = 12
    NUM_LAYERS = 6
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load dataset
    dataset = load_dataset("abisee/cnn_dailymail", "3.0.0")
    # train_set = dataset["train"].select(range(5000))
    # val_set = dataset["validation"].select(range(500))
    # test_set = dataset["test"].select(range(500))
    # dataset = DatasetDict({
    #     "train": train_set,
    #     "validation": val_set,
    #     "test": test_set
    # })

    # Train tokenizer
    tokenizer = CustomTokenizer()
    texts = [example["article"] + " " + example["highlights"] for example in dataset["train"]]
    tokenizer.train(texts, vocab_size=VOCAB_SIZE)
    
    # tokenizer = Tokenizer(models.BPE())
    # tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    # trainer = trainers.BpeTrainer(vocab_size=25000, special_tokens=["<pad>", "<sos>", "<eos>", "<unk>"])
    # tokenizer.train_from_iterator(texts, trainer=trainer)
    # tokenizer.model = models.BPE()
    # tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)
    # tokenizer.decoder = decoders.ByteLevel()
    # from transformers import PreTrainedTokenizerFast

    # tokenizer = PreTrainedTokenizerFast(
    #     tokenizer_object=tokenizer,
    #     bos_token="<sos>",
    #     eos_token="<eos>",
    #     unk_token="<unk>",
    #     pad_token="<pad>",
    # )


    # Create datasets
    train_dataset = SummarizationDataset(dataset["train"], tokenizer, MAX_SEQ_LEN, HEAD_LEN, TAIL_LEN)
    val_dataset = SummarizationDataset(dataset["validation"], tokenizer, MAX_SEQ_LEN, HEAD_LEN, TAIL_LEN)

    # DataLoader
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    # Initialize model, optimizer, and scaler
    model = DecoderOnlyTransformer(VOCAB_SIZE, MODEL_DIM, NUM_HEADS, NUM_LAYERS, MAX_SEQ_LEN).to(DEVICE)
    # model = DecoderOnlyModel(vocab_size=VOCAB_SIZE, d_model=256, nhead=4, num_layers=4, max_seq_length=1000).to(DEVICE)


    # Optimizer and Scheduler
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)


    # Run training
    # for epoch in range(EPOCHS):
    #     avg_loss = train(model, train_loader, optimizer, scaler, DEVICE)
    #     print(f"Epoch {epoch + 1}/{EPOCHS}, Loss: {avg_loss:.4f}")
    # Training Loop
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for step, batch in enumerate(tqdm(train_loader)):
            input_ids = batch["input_ids"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(input_ids)
            loss = nn.CrossEntropyLoss()(outputs.view(-1, outputs.size(-1)), labels.view(-1))
            loss.backward()
            optimizer.step()
            
            # scaler.scale(loss).backward()
            # scaler.step(optimizer)
            # scaler.update()
            # loss = loss / GRADIENT_ACCUMULATION_STEPS

            # if (step + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
            #     scaler.step(optimizer)
            #     scaler.update()
            #     optimizer.zero_grad()
            #     lr_scheduler.step()

            total_loss += loss.item()

            if (step + 1) % 100 == 0:
                print(f"Epoch {epoch + 1}, Step {step}, Loss: {total_loss}")

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(DEVICE)
                labels = batch["labels"].to(DEVICE)
                outputs = model(input_ids)
                loss = nn.CrossEntropyLoss()(outputs.view(-1, outputs.size(-1)), labels.view(-1))
                # loss = loss / GRADIENT_ACCUMULATION_STEPS
                val_loss += loss.item()

        print(f"Epoch {epoch + 1}, Validation Loss: {val_loss / len(val_loader)}")

    # Save Model
    save_dir = "custom_model_checkpoint"
    os.makedirs(save_dir, exist_ok=True)
    # config_path = f"{save_dir}/config.json"
    # with open(config_path, "w") as f:
    #     json.dump(model.config, f, indent=4)
    # print(f"Configuration saved at {config_path}")

    # Save model weights in multiple formats
    torch.save(model.state_dict(), f"{save_dir}/model.pth")  # Standard PyTorch format
    torch.save(model.state_dict(), f"{save_dir}/model.pt")   # Alternate extension
    torch.save(model.state_dict(), f"{save_dir}/pytorch_model.bin")  # Hugging Face-style

    print("Model weights saved in .pth, .pt, and .bin formats!")
    
if __name__ == "__main__":
    main()