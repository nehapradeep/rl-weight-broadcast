#!/bin/bash
# Quick Start Script for Weight Transfer System
# For 4 nodes with 8 AMD GPUs each

set -e

echo "=================================="
echo "Weight Transfer System Quick Start"
echo "=================================="
echo ""

# Step 1: Check environment
echo "Step 1: Checking environment..."
echo "  - Python version: $(python --version)"
echo "  - PyTorch version: $(python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'NOT INSTALLED')"
echo "  - CUDA/ROCm available: $(python -c 'import torch; print(torch.cuda.is_available())' 2>/dev/null || echo 'UNKNOWN')"
echo "  - GPU count: $(python -c 'import torch; print(torch.cuda.device_count())' 2>/dev/null || echo '0')"
echo ""

# Step 2: Install dependencies
read -p "Install/update dependencies? (y/n) " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Step 2: Installing dependencies..."
    
    # For AMD GPUs with ROCm 5.7
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.7
    
    # Other dependencies
    pip install transformers datasets accelerate
    
    echo "  ✓ Dependencies installed"
else
    echo "Step 2: Skipping dependency installation"
fi
echo ""

# Step 3: Run simple demo
echo "Step 3: Running simple demo (single node)..."
echo "  This tests the routing table computation logic"
echo ""
read -p "Run demo? (y/n) " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    python simple_demo.py
else
    echo "  Skipping demo"
fi
echo ""

# Step 4: Distributed setup instructions
echo "Step 4: Distributed Training Setup"
echo "=================================="
echo ""
echo "To run on your 4-node AMD GPU cluster:"
echo ""
echo "1. Identify your master node IP address:"
echo "   MASTER_IP=\$(hostname -I | awk '{print \$1}')"
echo "   echo \"Master IP: \$MASTER_IP\""
echo ""
echo "2. On EACH node, set environment variables:"
echo "   export NCCL_SOCKET_IFNAME=eth0  # or your network interface"
echo "   export NCCL_IB_DISABLE=1  # if not using InfiniBand"
echo "   export ROCR_VISIBLE_DEVICES=0,1,2,3,4,5,6,7"
echo ""
echo "3. Launch on each node:"
echo ""
echo "   # Node 0 (Master - ranks 0-7):"
echo "   torchrun --nproc_per_node=8 \\"
echo "            --nnodes=4 \\"
echo "            --node_rank=0 \\"
echo "            --master_addr=\$MASTER_IP \\"
echo "            --master_port=29500 \\"
echo "            main_distributed.py"
echo ""
echo "   # Node 1 (ranks 8-15):"
echo "   torchrun --nproc_per_node=8 \\"
echo "            --nnodes=4 \\"
echo "            --node_rank=1 \\"
echo "            --master_addr=\$MASTER_IP \\"
echo "            --master_port=29500 \\"
echo "            main_distributed.py"
echo ""
echo "   # Node 2 (ranks 16-23 - Controller):"
echo "   torchrun --nproc_per_node=8 \\"
echo "            --nnodes=4 \\"
echo "            --node_rank=2 \\"
echo "            --master_addr=\$MASTER_IP \\"
echo "            --master_port=29500 \\"
echo "            main_distributed.py"
echo ""
echo "   # Node 3 (ranks 24-31 - Inference):"
echo "   torchrun --nproc_per_node=8 \\"
echo "            --nnodes=4 \\"
echo "            --node_rank=3 \\"
echo "            --master_addr=\$MASTER_IP \\"
echo "            --master_port=29500 \\"
echo "            main_distributed.py"
echo ""

# Step 5: Create launch scripts
read -p "Generate launch scripts for each node? (y/n) " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Step 5: Generating launch scripts..."
    
    # Get master IP
    read -p "Enter master node IP address: " MASTER_IP
    
    for node_rank in 0 1 2 3; do
        cat > launch_node${node_rank}.sh << EOF
#!/bin/bash
# Launch script for Node ${node_rank}

# Set environment variables
export NCCL_SOCKET_IFNAME=eth0  # Adjust if needed
export NCCL_IB_DISABLE=1
export ROCR_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NCCL_DEBUG=INFO  # For debugging

# Launch training
torchrun --nproc_per_node=8 \\
         --nnodes=4 \\
         --node_rank=${node_rank} \\
         --master_addr=${MASTER_IP} \\
         --master_port=29500 \\
         main_distributed.py

EOF
        chmod +x launch_node${node_rank}.sh
        echo "  ✓ Created launch_node${node_rank}.sh"
    done
    
    echo ""
    echo "Launch scripts created! To run on each node:"
    echo "  Node 0: ./launch_node0.sh"
    echo "  Node 1: ./launch_node1.sh"
    echo "  Node 2: ./launch_node2.sh"
    echo "  Node 3: ./launch_node3.sh"
else
    echo "Step 5: Skipping script generation"
fi
echo ""

# Step 6: Testing single node
echo "Step 6: Single Node Testing"
echo "============================"
echo ""
echo "Before running on 4 nodes, test on a single node:"
echo ""
echo "  torchrun --nproc_per_node=8 \\"
echo "           --nnodes=1 \\"
echo "           main_distributed.py"
echo ""
read -p "Run single-node test now? (y/n) " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    torchrun --nproc_per_node=8 --nnodes=1 main_distributed.py
else
    echo "  Skipping single-node test"
fi
echo ""

echo "=================================="
echo "Quick Start Complete!"
echo "=================================="
echo ""
echo "Summary of your setup:"
echo "  - 4 nodes × 8 AMD GPUs = 32 total GPUs"
echo "  - Nodes 0-1: Training (16 GPUs)"
echo "  - Node 2: Controller (8 GPUs)"  
echo "  - Node 3: Inference (8 GPUs)"
echo ""
echo "Files created:"
echo "  - weight_transfer_controller.py"
echo "  - training_worker.py"
echo "  - inference_worker.py"
echo "  - main_distributed.py"
echo "  - simple_demo.py"
echo "  - README.md"
if [ -f "launch_node0.sh" ]; then
echo "  - launch_node[0-3].sh"
fi
echo ""
echo "Next steps:"
echo "  1. Review README.md for detailed documentation"
echo "  2. Test on single node first"
echo "  3. Deploy to all 4 nodes"
echo "  4. Monitor performance and optimize"
echo ""
echo "For issues, check:"
echo "  - Network connectivity between nodes"
echo "  - NCCL configuration (NCCL_DEBUG=INFO)"
echo "  - GPU memory usage"
echo "  - Firewall settings for port 29500"
echo ""
