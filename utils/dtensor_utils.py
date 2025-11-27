"""
DTensor utility functions for analyzing layout and computing shard assignments.
"""
from __future__ import annotations
from typing import List, Tuple, Dict, Any, Optional
import torch
from torch.distributed._tensor import DTensor, Replicate, Shard
from torch.distributed.device_mesh import DeviceMesh


def mesh_coordinate_to_gpu_index(mesh_coordinate: List[int], mesh_shape: List[int]) -> int:
    """
    Convert mesh coordinates to a linear GPU index.
    
    Args:
        mesh_coordinate: List of coordinates [i, j, k, ...] in mesh dimensions
        mesh_shape: Shape of the mesh [A, B, C, ...]
    
    Returns:
        Linear GPU index
    
    Example:
        mesh_shape = [2, 4]  # 2x4 grid
        coordinate [1, 2] -> GPU index = 1 * 4 + 2 = 6
    """
    if len(mesh_coordinate) != len(mesh_shape):
        raise ValueError(f"Coordinate dimension {len(mesh_coordinate)} != mesh dimension {len(mesh_shape)}")
    
    gpu_idx = 0
    stride = 1
    # Compute from rightmost dimension to leftmost (row-major order)
    for i in range(len(mesh_shape) - 1, -1, -1):
        if mesh_coordinate[i] < 0 or mesh_coordinate[i] >= mesh_shape[i]:
            raise ValueError(f"Coordinate {mesh_coordinate[i]} out of range [0, {mesh_shape[i]})")
        gpu_idx += mesh_coordinate[i] * stride
        stride *= mesh_shape[i]
    
    return gpu_idx


def gpu_index_to_mesh_coordinate(gpu_idx: int, mesh_shape: List[int]) -> List[int]:
    """
    Convert linear GPU index to mesh coordinates.
    
    Args:
        gpu_idx: Linear GPU index
        mesh_shape: Shape of the mesh [A, B, C, ...]
    
    Returns:
        List of coordinates [i, j, k, ...]
    
    Example:
        mesh_shape = [2, 4]  # 2x4 grid
        GPU index 6 -> coordinate [1, 2]
    """
    # Calculate total number of GPUs
    total_gpus = 1
    for dim in mesh_shape:
        total_gpus *= dim
    
    if gpu_idx < 0 or gpu_idx >= total_gpus:
        raise ValueError(f"GPU index {gpu_idx} out of range [0, {total_gpus})")
    
    coordinate = []
    remaining = gpu_idx
    
    # Compute from rightmost dimension to leftmost
    for i in range(len(mesh_shape) - 1, -1, -1):
        dim_size = mesh_shape[i]
        coord = remaining % dim_size
        coordinate.insert(0, coord)
        remaining = remaining // dim_size
    
    return coordinate


def compute_shard_slice_for_dimension(
    tensor_dim_size: int,
    num_shards: int,
    shard_index: int
) -> Tuple[int, int]:
    """
    Compute the slice (start, end) for a shard along one dimension.
    
    Args:
        tensor_dim_size: Size of the tensor dimension being sharded
        num_shards: Number of shards to split into
        shard_index: Which shard (0 to num_shards-1)
    
    Returns:
        Tuple of (start_index, end_index) for this shard
    
    Example:
        dim_size=100, num_shards=4, shard_index=1
        -> (25, 50)  # shard 1 gets rows 25-49
    """
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError(f"Shard index {shard_index} out of range [0, {num_shards})")
    
    # Calculate base shard size and remainder
    base_shard_size = tensor_dim_size // num_shards
    remainder = tensor_dim_size % num_shards
    
    # First 'remainder' shards get one extra element
    if shard_index < remainder:
        shard_size = base_shard_size + 1
        start = shard_index * shard_size
    else:
        shard_size = base_shard_size
        start = remainder * (base_shard_size + 1) + (shard_index - remainder) * base_shard_size
    
    end = start + shard_size
    return (start, end)


def compute_gpu_shard_assignments(
    dtensor: DTensor,
    mesh: DeviceMesh,
    placements: Tuple,
    global_shape: Tuple[int, ...]
) -> List[Dict[str, Any]]:
    """
    Compute which GPU holds which shard for a DTensor.
    Handles multi-dimensional meshes and multiple placements.
    
    Args:
        dtensor: The DTensor to analyze
        mesh: DeviceMesh object
        placements: Tuple of Placement objects
        global_shape: Global shape of the tensor
    
    Returns:
        List of dictionaries, one per GPU, containing:
        - gpu_index: Linear GPU index
        - mesh_coordinate: [i, j, k, ...] coordinates in mesh
        - device: Device string (e.g., "cuda:0")
        - shard_slices: Dict mapping tensor dimension -> (start, end) slice
        - shard_shape: Shape of the shard this GPU holds
        - size_bytes: Size of shard in bytes
    """
    mesh_shape = list(mesh.shape)
    devices = mesh.devices
    num_mesh_dims = len(mesh_shape)
    num_placements = len(placements)
    
    if num_mesh_dims != num_placements:
        raise ValueError(
            f"Mesh dimensions {num_mesh_dims} != placements {num_placements}. "
            f"Each mesh dimension must have a corresponding placement."
        )
    
    # Calculate total number of GPUs
    total_gpus = 1
    for dim_size in mesh_shape:
        total_gpus *= dim_size
    
    gpu_assignments = []
    
    # Iterate through all possible mesh coordinates
    for gpu_idx in range(total_gpus):
        # Convert GPU index to mesh coordinates
        mesh_coord = gpu_index_to_mesh_coordinate(gpu_idx, mesh_shape)
        device = devices[gpu_idx] if gpu_idx < len(devices) else None
        
        # Compute shard slices for each tensor dimension
        shard_slices = {}  # tensor_dim -> (start, end)
        shard_shape = list(global_shape)  # Start with global shape, will be modified
        
        # Process each mesh dimension and its corresponding placement
        for mesh_dim_idx, placement in enumerate(placements):
            mesh_dim_size = mesh_shape[mesh_dim_idx]
            coord_in_mesh_dim = mesh_coord[mesh_dim_idx]
            
            if isinstance(placement, Shard):
                # This mesh dimension shards a tensor dimension
                tensor_dim = placement.dim
                if tensor_dim >= len(global_shape):
                    continue  # Skip invalid dimension
                
                tensor_dim_size = global_shape[tensor_dim]
                
                # Compute shard slice for this dimension
                shard_start, shard_end = compute_shard_slice_for_dimension(
                    tensor_dim_size, mesh_dim_size, coord_in_mesh_dim
                )
                
                shard_slices[tensor_dim] = (shard_start, shard_end)
                shard_shape[tensor_dim] = shard_end - shard_start
                
            elif isinstance(placement, Replicate):
                # This mesh dimension replicates - all GPUs get full slice
                # No change to shard_shape for this dimension
                pass
            # Note: Partial() placement could be handled here if needed
        
        # Calculate shard size in bytes
        shard_numel = 1
        for dim_size in shard_shape:
            shard_numel *= dim_size
        
        element_size = dtensor.element_size() if hasattr(dtensor, "element_size") else None
        size_bytes = shard_numel * element_size if element_size else None
        
        gpu_assignments.append({
            "gpu_index": gpu_idx,
            "mesh_coordinate": mesh_coord,
            "device": str(device) if device is not None else f"unknown_{gpu_idx}",
            "shard_slices": shard_slices,  # Dict: tensor_dim -> (start, end)
            "shard_shape": shard_shape,
            "size_bytes": size_bytes,
        })
    
    return gpu_assignments


def get_current_gpu_shard_info(
    dtensor: DTensor,
    mesh: DeviceMesh
) -> Optional[Dict[str, Any]]:
    """
    Get shard information for the current GPU (the one running this code).
    
    Args:
        dtensor: The DTensor to analyze
        mesh: DeviceMesh object
    
    Returns:
        Dictionary with shard info for current GPU, or None if not available
    """
    if not isinstance(dtensor, DTensor):
        return None
    
    try:
        # Try to get current mesh coordinate
        if hasattr(mesh, "get_coordinate"):
            current_coord = mesh.get_coordinate()
            if current_coord is not None:
                current_coord = list(current_coord)
                mesh_shape = list(mesh.shape)
                gpu_idx = mesh_coordinate_to_gpu_index(current_coord, mesh_shape)
                
                # Get all GPU assignments and find current one
                placements = dtensor.placements if hasattr(dtensor, "placements") else None
                global_shape = dtensor.shape
                
                if placements is not None:
                    all_assignments = compute_gpu_shard_assignments(
                        dtensor, mesh, placements, global_shape
                    )
                    
                    for assignment in all_assignments:
                        if assignment["gpu_index"] == gpu_idx:
                            assignment["is_current_gpu"] = True
                            return assignment
    except Exception:
        pass
    
    return None

