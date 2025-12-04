"""
Configuration file for Weight Transfer System
Adjust these settings for your specific setup
"""

from dataclasses import dataclass
from typing import List


@dataclass
class ClusterConfig:
    """Cluster configuration"""
    # Number of nodes
    num_nodes: int = 4
    
    # GPUs per node
    gpus_per_node: int = 8
    
    # Total world size (automatically calculated)
    @property
    def world_size(self) -> int:
        return self.num_nodes * self.gpus_per_node
    
    # Network interface (for NCCL)
    network_interface: str = "ens26np0"  # Change to "ib0" for InfiniBand
    
    # Master node address
    master_addr: str = "45.76.29.254"  # Change to actual IP in multi-node setup
    master_port: int = 29500


@dataclass
class RoleConfig:
    """Role assignment configuration"""
    # Training ranks (nodes 0-1, ranks 0-15)
    training_start_rank: int = 0
    training_end_rank: int = 16
    
    # Controller rank (node 2, rank 16)
    controller_rank: int = 16
    
    # Inference ranks (node 3, ranks 24-31)
    inference_start_rank: int = 24
    inference_end_rank: int = 32
    
    @property
    def training_ranks(self) -> List[int]:
        return list(range(self.training_start_rank, self.training_end_rank))
    
    @property
    def inference_ranks(self) -> List[int]:
        return list(range(self.inference_start_rank, self.inference_end_rank))


@dataclass
class ModelConfig:
    """Model configuration"""
    # Model name or config
    model_name: str = "gpt2"  # Can be "gpt2", "gpt2-medium", "gpt2-large", etc.
    
    # Or use custom config
    use_custom_config: bool = False
    n_embd: int = 768
    n_layer: int = 12
    n_head: int = 12
    vocab_size: int = 50257
    
    # Precision
    use_fp16: bool = False
    use_bf16: bool = True


@dataclass
class DataConfig:
    """Data configuration"""
    # Dataset
    dataset_name: str = "wikitext"
    dataset_config: str = "wikitext-2-raw-v1"
    
    # Training parameters
    batch_size: int = 4
    max_length: int = 128
    
    # DataLoader
    num_workers: int = 4
    pin_memory: bool = True


@dataclass
class TrainingConfig:
    """Training configuration"""
    # Optimization
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    warmup_steps: int = 100
    max_steps: int = 10000
    
    # Gradient settings
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    
    # Logging
    log_interval: int = 10
    eval_interval: int = 500
    
    # Weight transfer frequency (in steps)
    weight_update_frequency: int = 100


@dataclass
class TransferConfig:
    """Weight transfer configuration"""
    # Pipelining
    enable_pipelining: bool = True
    pipeline_batch_size: int = 4
    
    # Memory management
    max_tmp_bytes: int = 2 << 30  # 2 GB
    
    # Communication
    communication_backend: str = "nccl"  # or "gloo" for CPU testing
    
    # Timeout (seconds)
    timeout_seconds: int = 1800
    
    # Verification
    verify_transfers: bool = True
    verify_sample_rate: float = 0.1  # Verify 10% of transfers


@dataclass
class SystemConfig:
    """Complete system configuration"""
    cluster: ClusterConfig = ClusterConfig()
    roles: RoleConfig = RoleConfig()
    model: ModelConfig = ModelConfig()
    data: DataConfig = DataConfig()
    training: TrainingConfig = TrainingConfig()
    transfer: TransferConfig = TransferConfig()
    
    def __post_init__(self):
        """Validate configuration"""
        # Check that roles don't overlap
        training_set = set(self.roles.training_ranks)
        inference_set = set(self.roles.inference_ranks)
        
        if training_set & inference_set:
            raise ValueError("Training and inference ranks overlap!")
        
        # Check that all ranks are within world size
        all_ranks = training_set | inference_set | {self.roles.controller_rank}
        if max(all_ranks) >= self.cluster.world_size:
            raise ValueError(f"Rank {max(all_ranks)} exceeds world size {self.cluster.world_size}")
        
        # Print configuration summary
        self.print_summary()
    
    def print_summary(self):
        """Print configuration summary"""
        print("\n" + "=" * 80)
        print(" " * 30 + "CONFIGURATION SUMMARY")
        print("=" * 80)
        
        print("\n📊 Cluster Configuration:")
        print(f"   Nodes: {self.cluster.num_nodes}")
        print(f"   GPUs per node: {self.cluster.gpus_per_node}")
        print(f"   Total GPUs: {self.cluster.world_size}")
        print(f"   Network: {self.cluster.network_interface}")
        
        print("\n🎭 Role Assignment:")
        print(f"   Training ranks: {self.roles.training_start_rank}-{self.roles.training_end_rank-1} ({len(self.roles.training_ranks)} GPUs)")
        print(f"   Controller rank: {self.roles.controller_rank}")
        print(f"   Inference ranks: {self.roles.inference_start_rank}-{self.roles.inference_end_rank-1} ({len(self.roles.inference_ranks)} GPUs)")
        
        print("\n🤖 Model Configuration:")
        print(f"   Model: {self.model.model_name}")
        print(f"   Precision: {'FP16' if self.model.use_fp16 else 'BF16' if self.model.use_bf16 else 'FP32'}")
        
        print("\n📚 Data Configuration:")
        print(f"   Dataset: {self.data.dataset_name} / {self.data.dataset_config}")
        print(f"   Batch size: {self.data.batch_size}")
        print(f"   Max length: {self.data.max_length}")
        
        print("\n🏋️ Training Configuration:")
        print(f"   Learning rate: {self.training.learning_rate}")
        print(f"   Max steps: {self.training.max_steps}")
        print(f"   Weight update frequency: {self.training.weight_update_frequency} steps")
        
        print("\n🔄 Transfer Configuration:")
        print(f"   Pipelining: {'Enabled' if self.transfer.enable_pipelining else 'Disabled'}")
        print(f"   Pipeline batch size: {self.transfer.pipeline_batch_size}")
        print(f"   Max temp memory: {self.transfer.max_tmp_bytes / 1e9:.1f} GB")
        print(f"   Backend: {self.transfer.communication_backend}")
        
        print("\n" + "=" * 80 + "\n")


# Default configuration
DEFAULT_CONFIG = SystemConfig()


# Example: Custom configuration for larger setup
def get_large_cluster_config() -> SystemConfig:
    """Configuration for larger cluster (8 nodes, 64 GPUs)"""
    config = SystemConfig()
    config.cluster.num_nodes = 8
    config.cluster.gpus_per_node = 8
    
    # Adjust roles
    config.roles.training_start_rank = 0
    config.roles.training_end_rank = 48  # 6 nodes for training
    config.roles.controller_rank = 48
    config.roles.inference_start_rank = 56
    config.roles.inference_end_rank = 64  # 1 node for inference
    
    # Use larger model
    config.model.model_name = "gpt2-large"
    
    # Larger batch size
    config.data.batch_size = 8
    
    return config


# Example: Configuration for testing on single node
def get_single_node_config() -> SystemConfig:
    """Configuration for testing on single node"""
    config = SystemConfig()
    config.cluster.num_nodes = 1
    config.cluster.gpus_per_node = 8
    
    # All roles on same node
    config.roles.training_start_rank = 0
    config.roles.training_end_rank = 4  # 4 GPUs for training
    config.roles.controller_rank = 4
    config.roles.inference_start_rank = 5
    config.roles.inference_end_rank = 8  # 3 GPUs for inference
    
    # Smaller model for testing
    config.model.use_custom_config = True
    config.model.n_embd = 256
    config.model.n_layer = 4
    config.model.n_head = 4
    
    # Smaller batch size
    config.data.batch_size = 2
    
    # Faster updates for testing
    config.training.weight_update_frequency = 10
    config.training.max_steps = 100
    
    return config


if __name__ == "__main__":
    print("\n=== Default Configuration ===")
    default = DEFAULT_CONFIG
    
    print("\n=== Single Node Configuration ===")
    single = get_single_node_config()
    
    print("\n=== Large Cluster Configuration ===")
    large = get_large_cluster_config()
