from datasets import load_dataset, DatasetDict
import torch
from torch.utils.data import DataLoader
from model_custom import DecoderOnlyTransformer, SummarizationDataset, CustomTokenizer
from tokenizers import processors
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 8
LEARNING_RATE = 5e-5
EPOCHS = 3
MAX_SEQ_LEN = 1000
HEAD_LEN = 700
TAIL_LEN = 300
VOCAB_SIZE = 50257  # Adjust based on your tokenizer
MODEL_DIM = 768
NUM_HEADS = 12
NUM_LAYERS = 6
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load dataset
dataset = load_dataset("abisee/cnn_dailymail","3.0.0")
# train_set = dataset["train"]#.select(range(1000))
# val_set = dataset["validation"]#.select(range(100))
test_set = dataset["test"]#.select(range(100))
# dataset = DatasetDict({
#     "train": train_set,
#     "validation": val_set,
#     "test": test_set
# })

# Train tokenizer
# tokenizer = CustomTokenizer()
texts = [train_set["article"] + " " + train_set["highlights"] for train_set in dataset["train"]]
# tokenizer.train(texts, vocab_size=VOCAB_SIZE)

tokenizer = Tokenizer(models.BPE())
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
trainer = trainers.BpeTrainer(vocab_size=25000, special_tokens=["<pad>", "<sos>", "<eos>", "<unk>"])
tokenizer.train_from_iterator(texts, trainer=trainer)
tokenizer.model = models.BPE()
tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)
tokenizer.decoder = decoders.ByteLevel()
from transformers import PreTrainedTokenizerFast

tokenizer = PreTrainedTokenizerFast(
    tokenizer_object=tokenizer,
    bos_token="<sos>",
    eos_token="<eos>",
    unk_token="<unk>",
    pad_token="<pad>",
)

# Create datasets
test_dataset = SummarizationDataset(dataset["test"], tokenizer, MAX_SEQ_LEN, HEAD_LEN, TAIL_LEN)


# DataLoader
test_loader = DataLoader(test_dataset, shuffle=False, batch_size=BATCH_SIZE)


save_path = "custom_model_checkpoint"

model = DecoderOnlyTransformer(VOCAB_SIZE, MODEL_DIM, NUM_HEADS, NUM_LAYERS, MAX_SEQ_LEN).to(DEVICE)
model.load_state_dict(torch.load(f"{save_path}/model.pth"))
model.eval()

# Prediction
accu_summaries = []
with torch.no_grad():
    for batch in test_loader:
        input_ids = batch["input_ids"].to(DEVICE)
        outputs = model(input_ids)
        token_ids = torch.argmax(outputs, dim=-1)
        decoded_outputs = [tokenizer.decode(pred.tolist()) for pred in token_ids]
        accu_summaries.extend(decoded_outputs)
        # print(len(decoded_outputs))
        # print(decoded_outputs[0])
        # print(len(test_set["highlights"]))


with open("prediction/summaries_small.txt.src", "w") as f:
    for content in test_set["article"]:
        content = content.replace("\n"," ")
        f.write(content + "\n")

with open("prediction/summaries_small_gold.txt.tgt", "w") as f:
    for content in test_set["highlights"]:
        content = content.replace("\n"," ")
        f.write(content + "\n")

with open("prediction/summaries_small_pred.txt.tgt", "w", encoding="utf-8") as f:
    for summary in accu_summaries:
        f.write(summary + "\n")  # separate summaries with a blank line
