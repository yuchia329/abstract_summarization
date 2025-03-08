import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers.optimization import AdamW
from torch.cuda.amp import GradScaler, autocast
import numpy as np
import os

# Custom Rotary Positional Encoding (RoPE)
class RotaryPositionalEncoding(nn.Module):
    def __init__(self, dim, max_seq_len):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, x):
        seq_len = x.size(1)
        t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
        return x * emb.cos() + self.rotate_half(x) * emb.sin()

    def rotate_half(self, x):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)

# Decoder-only Transformer with RoPE
class DecoderOnlyTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_layers, max_seq_len):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.rope = RotaryPositionalEncoding(d_model, max_seq_len)
        self.transformer = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model, nhead, batch_first=True),
            num_layers=num_layers,
        )
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        x = self.rope(x)
        x = self.transformer(x, x)
        return self.fc_out(x)

# Dataset class for CNN/DailyMail
class CNNDailyMailDataset(Dataset):
    def __init__(self, dataset, tokenizer, max_seq_len):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        article = self.dataset[idx]["article"]
        summary = self.dataset[idx]["highlights"]
        inputs = self.tokenizer.encode(article, max_length=self.max_seq_len, truncation=True, padding="max_length")
        labels = self.tokenizer.encode(summary, max_length=self.max_seq_len, truncation=True, padding="max_length")
        return torch.tensor(inputs), torch.tensor(labels)

# Training function
def train(model, dataloader, optimizer, scaler, device):
    model.train()
    total_loss = 0
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()

        with autocast():
            outputs = model(inputs)
            loss = nn.CrossEntropyLoss()(outputs.view(-1, outputs.size(-1)), labels.view(-1))

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
    return total_loss / len(dataloader)

# Main function
def main():
    # Hyperparameters
    vocab_size = 50257  # GPT-2 vocab size
    d_model = 768
    nhead = 12
    num_layers = 6
    max_seq_len = 512
    batch_size = 8
    epochs = 5
    lr = 5e-5
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load dataset
    dataset = load_dataset("abisee/cnn_dailymail", "3.0.0")
    train_dataset = CNNDailyMailDataset(dataset["train"], GPT2Tokenizer.from_pretrained("gpt2"), max_seq_len)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Initialize model, optimizer, and scaler
    model = DecoderOnlyTransformer(vocab_size, d_model, nhead, num_layers, max_seq_len).to(device)
    optimizer = AdamW(model.parameters(), lr=lr)
    scaler = GradScaler()

    # Training loop
    for epoch in range(epochs):
        avg_loss = train(model, train_dataloader, optimizer, scaler, device)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")

    # Save the model
    torch.save(model.state_dict(), "decoder_only_transformer_rope.pth")

if __name__ == "__main__":
    main()