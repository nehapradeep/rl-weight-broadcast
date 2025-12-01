"""
Simple Demo: Weight Transfer on Single Node
Tests the weight transfer concept without requiring multi-node setup

Usage:
    python simple_demo.py
"""

import torch
from transformers import GPT2LMHeadModel, GPT2Config
from weight_transfer_controller import (
    WeightTransferController,
    simulate_parameter_distribution
)


def demo_routing_computation():
    """Demonstrate routing table computation"""
    print("=" * 80)
    print("DEMO: Weight Transfer Routing Table Computation")
    print("=" * 80)
    print()
    
    # Create a small GPT-2 model
    print("Step 1: Creating GPT-2 model...")
    config = GPT2Config(
        n_embd=256,
        n_layer=4,
        n_head=4,
        vocab_size=50257
    )
    model = GPT2LMHeadModel(config)
    
    num_params = sum(p.numel() for p in model.parameters())
    param_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1e6
    print(f"   Model size: {num_params/1e6:.2f}M parameters ({param_size_mb:.2f} MB)")
    print()
    
    # Setup: Simulate 16 training GPUs and 8 inference GPUs
    training_ranks = list(range(0, 16))
    inference_ranks = list(range(16, 24))
    world_size = 24
    
    print("Step 2: Setting up distributed configuration...")
    print(f"   Training ranks: {training_ranks}")
    print(f"   Inference ranks: {inference_ranks}")
    print(f"   Total ranks: {world_size}")
    print()
    
    # Simulate parameter distribution
    print("Step 3: Simulating parameter distribution...")
    trainer_params, inference_params = simulate_parameter_distribution(
        model, training_ranks, inference_ranks, use_data_parallel=True
    )
    print(f"   Training: {len(training_ranks)} ranks with full model (Data Parallel)")
    print(f"   Inference: {len(inference_ranks)} ranks with full model")
    print()
    
    # Create controller
    print("Step 4: Creating Weight Transfer Controller...")
    controller = WeightTransferController(
        training_ranks=training_ranks,
        inference_ranks=inference_ranks,
        world_size=world_size
    )
    print()
    
    # Compute routing tables
    print("Step 5: Computing routing tables...")
    print("-" * 80)
    routing_tables = controller.compute_routing_tables(
        trainer_params, inference_params
    )
    print("-" * 80)
    print()
    
    # Analyze routing
    print("Step 6: Analyzing routing plan...")
    total_transfers = sum(len(table.transfers) for table in routing_tables.values())
    total_bytes = sum(table.total_bytes for table in routing_tables.values())
    
    print(f"   Total transfers: {total_transfers}")
    print(f"   Total data: {total_bytes/1e6:.2f} MB")
    print(f"   Average per training rank: {total_bytes/len(training_ranks)/1e6:.2f} MB")
    print()
    
    # Show load balancing
    print("Step 7: Load balancing analysis...")
    bytes_per_rank = {rank: table.total_bytes for rank, table in routing_tables.items()}
    min_bytes = min(bytes_per_rank.values())
    max_bytes = max(bytes_per_rank.values())
    imbalance = (max_bytes - min_bytes) / min_bytes * 100
    
    print(f"   Min bytes per rank: {min_bytes/1e6:.2f} MB")
    print(f"   Max bytes per rank: {max_bytes/1e6:.2f} MB")
    print(f"   Imbalance: {imbalance:.1f}%")
    print()
    
    # Show detailed routing for first two ranks
    print("Step 8: Sample routing table details...")
    print("-" * 80)
    for rank in sorted(routing_tables.keys())[:2]:
        table = routing_tables[rank]
        print(f"\nTraining Rank {rank}:")
        print(f"  Total transfers: {len(table.transfers)}")
        print(f"  Total data: {table.total_bytes/1e6:.2f} MB")
        print(f"  Sample transfers:")
        for entry in table.transfers[:3]:
            print(f"    - {entry.param_name:40s} -> Rank {entry.dst_rank}")
        if len(table.transfers) > 3:
            print(f"    ... and {len(table.transfers) - 3} more")
    print("-" * 80)
    print()
    
    # Performance estimate
    print("Step 9: Performance estimation...")
    # Assume 1 GB/s effective bandwidth per link
    bandwidth_per_link = 1e9  # bytes per second
    estimated_time = max_bytes / bandwidth_per_link
    print(f"   Estimated transfer time: {estimated_time:.2f} seconds")
    print(f"   (Assuming {bandwidth_per_link/1e9:.0f} GB/s effective bandwidth)")
    print()
    
    print("=" * 80)
    print("DEMO COMPLETED SUCCESSFULLY")
    print("=" * 80)
    print()
    print("Next steps:")
    print("  1. Run on actual distributed setup with: torchrun main_distributed.py")
    print("  2. Monitor actual transfer times and bandwidth")
    print("  3. Optimize based on profiling results")
    print()
    
    return routing_tables


def demo_weight_verification():
    """Demonstrate weight transfer verification"""
    print("=" * 80)
    print("DEMO: Weight Transfer Verification")
    print("=" * 80)
    print()
    
    print("Creating source and destination models...")
    config = GPT2Config(n_embd=128, n_layer=2, n_head=2)
    source_model = GPT2LMHeadModel(config)
    dest_model = GPT2LMHeadModel(config)
    
    # Initialize with different values
    for param in dest_model.parameters():
        param.data.fill_(0.0)
    
    print("Source model initialized with random weights")
    print("Destination model initialized with zeros")
    print()
    
    # Simulate weight transfer
    print("Simulating weight transfer...")
    transferred = 0
    total = 0
    for (src_name, src_param), (dst_name, dst_param) in zip(
        source_model.named_parameters(),
        dest_model.named_parameters()
    ):
        assert src_name == dst_name
        dst_param.data.copy_(src_param.data)
        transferred += 1
        total += src_param.numel()
    
    print(f"Transferred {transferred} parameters ({total/1e6:.2f}M elements)")
    print()
    
    # Verify
    print("Verifying transfer...")
    matches = 0
    for (src_name, src_param), (dst_name, dst_param) in zip(
        source_model.named_parameters(),
        dest_model.named_parameters()
    ):
        if torch.equal(src_param.data, dst_param.data):
            matches += 1
    
    print(f"Verification: {matches}/{transferred} parameters match")
    print(f"Success rate: {matches/transferred*100:.1f}%")
    print()
    
    # Test inference
    print("Testing inference with transferred weights...")
    source_model.eval()
    dest_model.eval()
    
    test_input = torch.randint(0, config.vocab_size, (1, 10))
    
    with torch.no_grad():
        src_output = source_model(test_input).logits
        dst_output = dest_model(test_input).logits
    
    output_diff = (src_output - dst_output).abs().max().item()
    print(f"Max output difference: {output_diff:.2e}")
    
    if output_diff < 1e-6:
        print("✓ Outputs match perfectly!")
    else:
        print("✗ Outputs differ (unexpected)")
    
    print()
    print("=" * 80)
    print("VERIFICATION COMPLETED")
    print("=" * 80)
    print()


def demo_parameter_sharding():
    """Demonstrate parameter sharding concepts"""
    print("=" * 80)
    print("DEMO: Parameter Sharding Concepts")
    print("=" * 80)
    print()
    
    print("This demo illustrates how parameters would be distributed")
    print("in different parallelism strategies:")
    print()
    
    # Create a simple model
    config = GPT2Config(n_embd=256, n_layer=2, n_head=4)
    model = GPT2LMHeadModel(config)
    
    print("Model layers:")
    for i, (name, param) in enumerate(list(model.named_parameters())[:10]):
        shape_str = "×".join(map(str, param.shape))
        print(f"  {i+1:2d}. {name:50s} [{shape_str:15s}] {param.numel():>10,} params")
    print(f"  ... and {len(list(model.parameters())) - 10} more layers")
    print()
    
    # Data Parallel (DP)
    print("1. Data Parallel (DP):")
    print("   - Each GPU has the FULL model")
    print("   - Different GPUs process different data batches")
    print("   - All GPUs have identical parameters")
    print("   - Weight transfer: ANY training GPU can send to ANY inference GPU")
    print()
    
    # Fully Sharded Data Parallel (FSDP)
    print("2. Fully Sharded Data Parallel (FSDP):")
    print("   - Each GPU has a SHARD of the model")
    print("   - Parameters are partitioned across GPUs")
    print("   - During forward/backward: all-gather to reconstruct full parameters")
    print("   - Weight transfer: Need to gather from multiple training GPUs")
    print()
    
    # Pipeline Parallel (PP)
    print("3. Pipeline Parallel (PP):")
    print("   - Different GPUs have different LAYERS")
    print("   - GPU 0: layers 0-5, GPU 1: layers 6-11, etc.")
    print("   - Weight transfer: Each training GPU sends its layers")
    print()
    
    # Tensor Parallel (TP)
    print("4. Tensor Parallel (TP):")
    print("   - Each LAYER is split across GPUs")
    print("   - E.g., attention heads split across GPUs")
    print("   - Weight transfer: Need to gather from multiple GPUs per layer")
    print()
    
    print("Current Implementation:")
    print("  ✓ Supports Data Parallel (DP)")
    print("  ○ FSDP support can be added (blog post shows how)")
    print("  ○ PP/TP support requires additional routing logic")
    print()
    
    print("=" * 80)
    print("SHARDING CONCEPTS DEMO COMPLETED")
    print("=" * 80)
    print()


def main():
    """Run all demos"""
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 20 + "Weight Transfer System Demo" + " " * 31 + "║")
    print("╚" + "=" * 78 + "╝")
    print()
    
    demos = [
        ("Routing Computation", demo_routing_computation),
        ("Weight Verification", demo_weight_verification),
        ("Parameter Sharding", demo_parameter_sharding),
    ]
    
    for i, (name, demo_func) in enumerate(demos, 1):
        print(f"\n>>> Running Demo {i}/{len(demos)}: {name}\n")
        try:
            demo_func()
        except Exception as e:
            print(f"\n✗ Demo failed with error: {e}\n")
            import traceback
            traceback.print_exc()
            continue
        
        if i < len(demos):
            input("\nPress Enter to continue to next demo...")
            print("\n" * 2)
    
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 28 + "All Demos Completed" + " " * 31 + "║")
    print("╚" + "=" * 78 + "╝")
    print()
    print("To run the full distributed system:")
    print("  1. Review README.md for setup instructions")
    print("  2. Configure your 4-node cluster")
    print("  3. Run: torchrun --nproc_per_node=8 --nnodes=4 main_distributed.py")
    print()


if __name__ == "__main__":
    main()
