# abstract_summarization

## Execute
### Training
Use the below command to run training, specify the idle GPU on your device: \
```CUDA_VISIBLE_DEVICES=1 python main.py```

### Prediction
The trained model is saved in output folder. It contains many checkpoint folder throughout the training. Use the latest one for prediction
In ```prediction.py```, update ```save_path``` to the checkpoint folder you specify, remove ```.select(range(1000))``` from line 12 to 14

Run the command: \
```CUDA_VISIBLE_DEVICES=1 python prediction.py```

### Evaluate output
Update ```hyp_path``` and ```ref_path``` in ```evaluate.py``` if changed the output files name in the prediction step.

Run the command:
```python evaluation.py```

## Dataset
The dataset is **abisee/cnn_dailymail**, version 3.0.0
I pull data from huggingface instead of provided dataset. The data size are same so it should have the same training result

## Model
The first model I use is t5-base model with 223M parameters. I randomly initialize the model parameter so it fulfills the no-pretrain model requirement. I also use custom tokenizer for the same purpose.

**TODO** \
The second model is a decoder-only transformer. Since I self-define the training script. I truncate long sequence with head:tail ratio to 2:1. I assume tail contains essential information for summarization. I also use Rotary Positional Encoding and Flash attention in the decoder model. Since the dataset is large, we use mix-precision training to reduce training time.

## Takeaway
Using Data Parallel with 2 GPUs does not help reducing training time. Since GPU communication is crucial for Parallelism, I inspect the GPUs links and discover the communication protocols are very slow. With data parallel strategy, I need 4 GPUs to surpass the training speed on a single GPU. Other parallelism need more inter GPUs communication which slower the training speed even more. To reproduce the table, use the command:
```nvidia-smi topo -m``` 

| GPU  | GPU0 | GPU1 | GPU2 | GPU3 | GPU4 | GPU5 |
|------|------|------|------|------|------|------|
| GPU0 | X    | PIX  | NODE | NODE | SYS  | SYS  |
| GPU1 | PIX  | X    | NODE | NODE | SYS  | SYS  |
| GPU2 | NODE | NODE | X    | PIX  | SYS  | SYS  |
| GPU3 | NODE | NODE | PIX  | X    | SYS  | SYS  |
| GPU4 | SYS  | SYS  | SYS  | SYS  | X    | PIX  |
| GPU5 | SYS  | SYS  | SYS  | SYS  | PIX  | X    |

**Legend**
- **X**: Self-connection (same GPU)
- **PIX**: GPUs connected via PCIe with peer-to-peer access
- **NODE**: GPUs connected via the same node but with higher latency
- **SYS**: GPUs connected via system memory with higher latency

