# GPU Model Broadcasting with Distributed Training & Inference

A PyTorch-based system for broadcasting ML models across multiple GPUs using RDMA (Remote Direct Memory Access) via UCCL, followed by distributed training and inference operations.

## 🎯 Overview

This project demonstrates high-performance model distribution across 3 nodes:
- **Rank 0 (Broadcaster)**: Loads and broadcasts GPT-2 model to other nodes
- **Rank 1 (Training Node)**: Receives model and trains on WikiText-2 dataset
- **Rank 2 (Inference Node)**: Receives model and runs text generation inference

## ✨ Features

- 🚀 **High-speed GPU-to-GPU transfer** using RDMA
- 📊 **Real dataset training** on WikiText-2 (Wikipedia articles)
- 🔍 **Comprehensive logging** with timestamps, metrics, and progress tracking
- 📈 **Training metrics**: Loss, perplexity, gradient norms, throughput
- ⚡ **Inference metrics**: Tokens/second, generation time, bandwidth
- 💾 **Checkpoint saving** for trained models
- 🔧 **Configurable** batch size, learning rate, epochs, etc.

## 📋 Requirements

### Hardware
- 3+ NVIDIA GPUs (can be on different machines)
- RDMA-capable network (InfiniBand or RoCE)
- UCCL library installed

### Software
- Python 3.8+
- CUDA 11.0+
- PyTorch 2.0+
- See `requirements.txt` for Python dependencies

## 🛠️ Installation

### 1. Install Dependencies

```bash
# Install Python packages
pip install -r requirements.txt

# Or install individually:
pip install torch transformers datasets pytz pandas
```

### 2. Install UCCL

Follow UCCL installation instructions for your system. Ensure the `uccl` Python module is available.

### 3. Verify GPU Access

```bash
# Check GPUs are visible
nvidia-smi

# Test PyTorch CUDA
python -c "import torch; print(torch.cuda.is_available())"
```

## 🚀 Usage

### Quick Start (3 Nodes)

Run on 3 different machines or GPUs:

```bash
# Machine 0 - Broadcaster
torchrun --nproc_per_node=1 --nnodes=3 --node_rank=0 \
  --master_addr=192.168.1.100 --master_port=29500 \
  gpu_transfer_wikitext2.py

# Machine 1 - Training Node
torchrun --nproc_per_node=1 --nnodes=3 --node_rank=1 \
  --master_addr=192.168.1.100 --master_port=29500 \
  gpu_transfer_wikitext2.py

# Machine 2 - Inference Node
torchrun --nproc_per_node=1 --nnodes=3 --node_rank=2 \
  --master_addr=192.168.1.100 --master_port=29500 \
  gpu_transfer_wikitext2.py
```

**Parameters:**
- `--master_addr`: IP address of rank 0 machine
- `--master_port`: Any free port (default: 29500)
- `--node_rank`: Unique rank for each node (0, 1, 2)
- `--nnodes`: Total number of nodes (3)

### Single Machine (Multi-GPU)

```bash
# All 3 processes on one machine
torchrun --nproc_per_node=3 --nnodes=1 gpu_transfer_wikitext2.py
```

## 📊 What Each Node Does

### Rank 0: Broadcaster 📡
1. Loads GPT-2 model from Hugging Face
2. Establishes RDMA connections to Rank 1 & 2
3. Broadcasts all model weights via GPU-to-GPU transfer
4. Logs transfer speed and bandwidth
5. Exits after broadcast complete

**Key Metrics:**
- Transfer time
- Bandwidth (GB/s)
- Tensors transferred

### Rank 1: Training Node 🎓
1. Accepts connection from broadcaster
2. Receives model weights
3. Downloads WikiText-2 dataset (200 samples)
4. Trains for 2 epochs with batch size 4
5. Saves trained model checkpoint

**Training Details:**
- **Dataset**: WikiText-2 (Wikipedia articles)
- **Loss Function**: Cross-Entropy (Causal Language Modeling)
- **Optimizer**: AdamW (lr=5e-5)
- **Batch Size**: 4
- **Epochs**: 2
- **Metrics**: Loss, Perplexity, Gradient Norms

**Key Metrics:**
- Training loss per step
- Perplexity (exp(loss))
- Gradient norms
- Steps per second
- Loss improvement over epochs

### Rank 2: Inference Node 🔮
1. Accepts connection from broadcaster
2. Receives model weights
3. Runs text generation on 5 sample prompts
4. Logs generation speed and quality

**Inference Details:**
- **Generation Settings**: Temperature=0.7, top_k=50, top_p=0.95
- **Max Length**: 50 tokens
- **Prompts**: Pre-defined creative prompts

**Key Metrics:**
- Inference time per sample
- Tokens generated per second
- Generation latency

## 📁 Output Files

### Logs
All ranks create detailed logs in the `logs/` directory:

```
logs/
├── rank_0_20241114_174827.log  # Broadcaster logs
├── rank_1_20241114_174827.log  # Training logs
└── rank_2_20241114_174827.log  # Inference logs
```

**Log Contents:**
- Timestamps for every operation
- Transfer progress and bandwidth
- Training loss and perplexity
- Inference results and timing
- Error messages and debugging info

### Checkpoints
Trained models are saved to `checkpoints/`:

```
checkpoints/
└── model_rank1_wikitext2_trained.pt
```

**Checkpoint Contains:**
- Model state dict
- Optimizer state dict
- Training metrics (losses per epoch)
- Total training steps

## ⚙️ Configuration

You can modify training/inference settings by editing the code:

### Training Configuration
```python
# In main() function, rank == 1 section:
run_training(
    model, 
    tokenizer, 
    num_epochs=2,      # Number of training epochs
    batch_size=4,      # Batch size
    lr=5e-5           # Learning rate
)
```

### Dataset Configuration
```python
# In prepare_dataset() function:
prepare_dataset(
    tokenizer, 
    max_length=128,    # Maximum sequence length
    num_samples=200    # Number of training samples
)
```

### Inference Configuration
```python
# In main() function, rank == 2 section:
run_inference(
    model, 
    tokenizer, 
    num_samples=5      # Number of inference samples
)
```

## 🎨 Example Output

### Broadcaster (Rank 0)
```
2024-11-14 17:48:27 | Rank 0 | INFO | BROADCAST START - Sending to 2 receivers
2024-11-14 17:48:27 | Rank 0 | INFO | Total tensors: 148
2024-11-14 17:48:27 | Rank 0 | INFO | Total size: 548.12 MB
2024-11-14 17:48:32 | Rank 0 | INFO | BROADCAST COMPLETE
2024-11-14 17:48:32 | Rank 0 | INFO | Total time: 5.23s
2024-11-14 17:48:32 | Rank 0 | INFO | Average bandwidth: 1.05 GB/s
```

### Training Node (Rank 1)
```
2024-11-14 17:48:35 | Rank 1 | INFO | EPOCH 1/2
2024-11-14 17:48:36 | Rank 1 | INFO | Step 10/100 | Loss: 3.2456 | Perplexity: 25.67 | Time: 0.125s
2024-11-14 17:48:40 | Rank 1 | INFO | EPOCH 1 SUMMARY
2024-11-14 17:48:40 | Rank 1 | INFO | Average Loss: 2.9845
2024-11-14 17:48:40 | Rank 1 | INFO | Average Perplexity: 19.75
2024-11-14 17:48:45 | Rank 1 | INFO | TRAINING COMPLETE
2024-11-14 17:48:45 | Rank 1 | INFO | Final loss: 2.7234
2024-11-14 17:48:45 | Rank 1 | INFO | Model checkpoint saved
```

### Inference Node (Rank 2)
```
2024-11-14 17:48:35 | Rank 2 | INFO | SAMPLE 1/5
2024-11-14 17:48:35 | Rank 2 | INFO | Prompt: 'Once upon a time'
2024-11-14 17:48:36 | Rank 2 | INFO | Output: 'Once upon a time, there was a kingdom...'
2024-11-14 17:48:36 | Rank 2 | INFO | Tokens/sec: 64.3
2024-11-14 17:48:42 | Rank 2 | INFO | INFERENCE COMPLETE
2024-11-14 17:48:42 | Rank 2 | INFO | Average tokens/sec: 62.1
```

## 🔧 Troubleshooting

### Common Issues

#### 1. "No module named 'pytz'" or "No module named 'pandas'"
```bash
pip install pytz pandas datasets
```

#### 2. "Connect failed" or "Accept failed"
- Ensure all machines can reach each other
- Check firewall settings
- Verify `--master_addr` is correct
- Try a different `--master_port`

#### 3. CUDA Out of Memory
Reduce batch size in the code:
```python
run_training(model, tokenizer, batch_size=2)  # Reduce from 4 to 2
```

#### 4. "UCCL not found"
Ensure UCCL is properly installed and the Python module is accessible:
```bash
python -c "from uccl import p2p"
```

#### 5. Slow Transfer Speed
- Verify RDMA is working: `ibstat` or `ibv_devinfo`
- Check network configuration
- Ensure GPUDirect RDMA is enabled
- Monitor with: `watch -n 1 nvidia-smi`

### Network Configuration

Check if nodes can communicate:
```bash
# On rank 0 machine
ping <rank_1_ip>
ping <rank_2_ip>

# Check port availability
netstat -tuln | grep 29500
```

### Debugging

Enable debug logging by changing in the code:
```python
logging.basicConfig(level=logging.DEBUG, ...)  # Instead of INFO
```

## 📈 Performance Tips

1. **Increase batch size** if you have enough GPU memory
2. **Adjust learning rate** based on batch size (higher batch = higher lr)
3. **Use more training samples** for better model quality
4. **Enable mixed precision training** with `torch.cuda.amp`
5. **Monitor GPU utilization** with `nvidia-smi`

## 🔄 Extending the Code

### Add More Receivers
Modify the broadcaster section:
```python
if rank == 0:
    for receiver_rank in [1, 2, 3, 4]:  # Add more ranks
        # ... connection code
```

### Use Different Datasets
Replace in `prepare_dataset()`:
```python
# Instead of WikiText-2:
dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")

# Use other datasets:
dataset = load_dataset("openwebtext", split="train")  # Web text
dataset = load_dataset("squad", split="train")        # Q&A dataset
dataset = load_dataset("imdb", split="train")         # Movie reviews
```

### Change Model Architecture
Replace GPT-2 with other models:
```python
# Instead of GPT-2:
from transformers import GPT2LMHeadModel
model = GPT2LMHeadModel.from_pretrained("gpt2")

# Use other models:
from transformers import BertForMaskedLM
model = BertForMaskedLM.from_pretrained("bert-base-uncased")

from transformers import T5ForConditionalGeneration
model = T5ForConditionalGeneration.from_pretrained("t5-small")
```

## 📚 Technical Details

### Loss Function: Cross-Entropy

The training uses **Cross-Entropy Loss** for causal language modeling:
- Model predicts next token at each position
- Loss = negative log probability of correct next token
- Lower loss = better predictions
- Perplexity = exp(loss) - measures model "surprise"

### Transfer Protocol

1. **Metadata Exchange**: All nodes exchange connection info via PyTorch distributed
2. **Connection**: Broadcaster connects to each receiver
3. **Memory Registration**: Each tensor is registered for RDMA
4. **Transfer**: Direct GPU-to-GPU transfer without CPU involvement
5. **Verification**: Receivers copy data into model state dict

## 📝 Citation

If you use this code in your research, please cite:

```bibtex
@software{gpu_model_broadcast,
  title = {GPU Model Broadcasting with Distributed Training},
  author = {Your Name},
  year = {2024},
  url = {https://github.com/yourusername/yourrepo}
}
```

## 📄 License

MIT License - feel free to use and modify!

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 💬 Support

If you encounter issues:
1. Check the logs in `logs/` directory
2. Review the troubleshooting section above
3. Open an issue on GitHub with your logs

## 🎉 Acknowledgments

- **PyTorch** for distributed training framework
- **Hugging Face** for transformers and datasets
- **UCCL** for high-speed RDMA communication
- **WikiText-2** dataset authors

---

**Happy Broadcasting!** 🚀
