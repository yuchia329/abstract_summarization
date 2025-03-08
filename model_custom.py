import torch
import torch.nn as nn
from torch.utils.data import Dataset
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

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

# Dataset Class
class SummarizationDataset(Dataset):
    def __init__(self, dataset, tokenizer, max_seq_len, head_len, tail_len):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.head_len = head_len
        self.tail_len = tail_len

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        article = self.dataset[idx]["article"]
        highlights = self.dataset[idx]["highlights"]

        # Truncate article: 700 from head, 300 from tail
        tokens = self.tokenizer.encode(article, max_length=self.max_seq_len, padding="max_length")
        if len(tokens) > self.max_seq_len:
            head = tokens[:self.head_len]
            tail = tokens[-self.tail_len:]
            tokens = head + tail
            tokens = tokens[:self.max_seq_len]

        # Encode highlights
        highlights_ids = self.tokenizer.encode(highlights, max_length=self.max_seq_len, padding="max_length")

        return {
            "input_ids": torch.tensor(tokens, dtype=torch.long),
            "labels": torch.tensor(highlights_ids, dtype=torch.long),
        }

# Custom Tokenizer
class CustomTokenizer:
    def __init__(self):
        self.model = models.BPE()
        self.tokenizer = Tokenizer(models.BPE())
        self.tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        self.tokenizer.decoder = decoders.ByteLevel()
        self.pad_token = "<pad>"
        self.sos_token = "<sos>"
        self.eos_token = "<eos>"
        self.unk_token = "<unk>"

    def train(self, texts, vocab_size):
        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            special_tokens=[self.pad_token, self.sos_token, self.eos_token, self.unk_token]
        )
        self.tokenizer.train_from_iterator(texts, trainer=trainer)

    def encode(self, text, max_length=None, padding=False):
        # Encode the text
        encoding = self.tokenizer.encode(text)
        ids = encoding.ids

        # Truncate if max_length is provided
        if max_length is not None:
            ids = ids[:max_length]

        # Add padding if required
        if padding and max_length is not None:
            pad_id = self.tokenizer.token_to_id(self.pad_token)
            if len(ids) < max_length:
                ids.extend([pad_id] * (max_length - len(ids)))
        return ids
        
    def decode(self, ids):
        # Remove padding tokens before decoding
        pad_id = self.tokenizer.token_to_id(self.pad_token)
        ids = [token_id for token_id in ids if token_id != pad_id]
        return self.tokenizer.decode(ids)

    def get_vocab_size(self):
        return self.tokenizer.get_vocab_size()

    def token_to_id(self, token):
        return self.tokenizer.token_to_id(token)

    def id_to_token(self, id):
        return self.tokenizer.id_to_token(id)
