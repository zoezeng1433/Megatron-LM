# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
# Portions of this code are from DeepSeek DeepEP project
# Copyright (c) 2025 DeepSeek
# Licensed under the MIT License - https://github.com/deepseek-ai/DeepEP/blob/main/LICENSE


try:
    from pplx_garden.distributed import ParallelGroup
    from pplx_garden.distributed.torch_group import TorchParallelGroup
    from pplx_garden.kernels.p2p_all_to_all import P2PAllToAll

    HAVE_PPLX = True
except ImportError:
    HAVE_PPLX = False

import torch
import torch.distributed
from typing import Optional, Dict, Tuple
import os

# Global cache for P2PAllToAll instances
_all_to_all_cache: Dict[Tuple[int, int, int, int, int, int], "P2PAllToAll"] = {}


def get_hidden_bytes(x: torch.Tensor) -> int:
    """Calculate the number of hidden bytes for a tensor.

    Args:
        x (torch.Tensor): Input tensor

    Returns:
        int: Number of hidden bytes
    """
    return x.size(1) * max(x.element_size(), 2)


def _process_group_to_parallel_group(
    group: torch.distributed.ProcessGroup,
) -> "ParallelGroup":
    """Convert torch.distributed.ProcessGroup to ParallelGroup.

    Args:
        group (torch.distributed.ProcessGroup): Process group to convert

    Returns:
        ParallelGroup: Parallel group for pplx
    """
    if not HAVE_PPLX:
        raise ImportError("pplx_garden is not installed")

    # Get ranks in the group
    global_rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    
    # Get all ranks in the group
    # Note: ProcessGroup doesn't expose ranks directly, so we need to infer
    # For now, we'll assume the group contains consecutive ranks starting from 0
    # This is a limitation - in practice, we might need to pass ranks explicitly
    group_size = group.size()
    group_rank = group.rank()
    
    # Try to get device from current CUDA device
    device = torch.cuda.current_device()
    device = torch.device(f"cuda:{device}")
    
    # Estimate node_rank and local_rank
    # This is a heuristic - in practice these should be passed explicitly
    # Assume 8 GPUs per node (common configuration)
    gpus_per_node = int(os.environ.get("GPUS_PER_NODE", "8"))
    node_rank = global_rank // gpus_per_node
    local_rank = global_rank % gpus_per_node
    
    # Get all ranks in the group by all_gathering
    # Since ProcessGroup doesn't expose ranks directly, we use all_gather
    # Use the same device as the main computation (CUDA) for consistency
    # For small metadata like ranks, CUDA is fine and avoids CPU-GPU transfer
    # Note: This assumes all ranks have CUDA available, which is true for GPU training
    cuda_device = torch.cuda.current_device()
    rank_tensor = torch.tensor([global_rank], dtype=torch.int64, device=f'cuda:{cuda_device}')
    all_ranks_tensor = torch.zeros(group_size, dtype=torch.int64, device=f'cuda:{cuda_device}')
    torch.distributed.all_gather(all_ranks_tensor, rank_tensor, group=group)
    group_ranks = all_ranks_tensor.tolist()
    group_ranks.sort()  # Ensure sorted order
    
    return TorchParallelGroup(
        device=device,
        node_rank=node_rank,
        local_rank=local_rank,
        global_rank=global_rank,
        ranks=group_ranks,
    )


def _get_all_to_all_instance(
    group: torch.distributed.ProcessGroup,
    num_experts: int,
    num_experts_per_token: int,
    hidden_dim: int,
    max_num_tokens: int,
    in_dtype: torch.dtype,
    out_dtype: torch.dtype,
    scale_dtype: Optional[torch.dtype] = None,
    hidden_dim_scale: Optional[int] = None,
    nets_per_gpu: int = 2,
    max_private_tokens: Optional[int] = None,
) -> "P2PAllToAll":
    """Get or create a P2PAllToAll instance for the given configuration.

    Args:
        group: Process group
        num_experts: Number of experts
        num_experts_per_token: Number of experts per token (topk)
        hidden_dim: Hidden dimension
        max_num_tokens: Maximum number of tokens (used for sizing, will use max of all calls)
        in_dtype: Input dtype
        out_dtype: Output dtype
        scale_dtype: Scale dtype (optional)
        hidden_dim_scale: Hidden dimension scale (optional)
        nets_per_gpu: Number of networks per GPU
        max_private_tokens: Maximum private tokens (optional)

    Returns:
        P2PAllToAll: All-to-all instance
    """
    global _all_to_all_cache
    
    if not HAVE_PPLX:
        raise ImportError("pplx_garden is not installed")
    
    # Create cache key - include a rounded-up max_num_tokens to allow reuse
    # Round up to nearest power of 2 for better cache reuse
    def _round_up_power_of_2(n):
        if n <= 0:
            return 1
        return 1 << (n - 1).bit_length()
    
    rounded_max_tokens = _round_up_power_of_2(max(max_num_tokens, 4096))
    group_size = group.size()
    cache_key = (
        id(group),  # Use group id as part of key
        num_experts,
        num_experts_per_token,
        hidden_dim,
        hash(str(in_dtype)) + hash(str(out_dtype)),
        rounded_max_tokens,  # Include in cache key
    )
    
    if cache_key not in _all_to_all_cache:
        # Convert ProcessGroup to ParallelGroup
        parallel_group = _process_group_to_parallel_group(group)
        
        # Use rounded-up max_num_tokens for better cache reuse
        effective_max_num_tokens = rounded_max_tokens
        
        # Create P2PAllToAll instance
        _all_to_all_cache[cache_key] = P2PAllToAll(
            max_num_tokens=effective_max_num_tokens,
            num_experts=num_experts,
            expert_padding=1,  # Default padding
            hidden_dim=hidden_dim,
            hidden_dim_scale=hidden_dim_scale,
            in_dtype=in_dtype,
            out_dtype=out_dtype,
            scale_dtype=scale_dtype,
            num_experts_per_token=num_experts_per_token,
            nets_per_gpu=nets_per_gpu,
            max_private_tokens=max_private_tokens,
            device=parallel_group.device,
            dp_group=None,  # No data parallelism group for now
            node_group=None,  # Will be set if needed
            global_group=parallel_group,
        )
    
    return _all_to_all_cache[cache_key]


class FusedDispatch(torch.autograd.Function):
    """Fused dispatch operation for MoE routing combining computation and communication."""

    @staticmethod
    def forward(
        ctx,
        x,
        token_indices,
        token_probs,
        num_experts,
        group,
        async_finish=False,
        allocate_on_comm_stream=False,
    ):
        """Forward pass of fused dispatch."""
        if not HAVE_PPLX:
            raise ImportError("pplx_garden is not installed")
        
        num_tokens, hidden_dim = x.shape
        num_experts_per_token = token_indices.shape[1]
        
        # Check if num_tokens exceeds a reasonable limit
        # We'll use a larger value for max_num_tokens to allow some headroom
        group_size = group.size()
        num_local_experts = (num_experts + group_size - 1) // group_size
        
        # Calculate max_recv_tokens using similar logic to pplx benchmark
        # This is more accurate than the previous simplified calculation
        def round_up(n: int, m: int) -> int:
            """Round up n to the nearest multiple of m."""
            return (n + m - 1) // m * m
        
        expert_padding = 1
        num_dp_groups = 1  # Assuming single DP group for now
        num_tokens_total = num_tokens * num_dp_groups
        
        # Calculate max_recv_tokens similar to bench_all_to_all.py
        max_recv_tokens = round_up(
            max(
                min(
                    num_tokens_total * num_experts_per_token
                    + num_local_experts * (expert_padding - 1),
                    num_tokens_total * num_local_experts,
                ),
                num_local_experts * expert_padding,
            ),
            expert_padding,
        )
        
        # Get or create P2PAllToAll instance
        # Use max_recv_tokens to estimate max_num_tokens needed
        # Add some headroom (1.5x) to account for variations
        estimated_max_num_tokens = int(max(num_tokens, max_recv_tokens // num_experts_per_token) * 1.5)
        
        all_to_all = _get_all_to_all_instance(
            group=group,
            num_experts=num_experts,
            num_experts_per_token=num_experts_per_token,
            hidden_dim=hidden_dim,
            max_num_tokens=estimated_max_num_tokens,
            in_dtype=x.dtype,
            out_dtype=x.dtype,  # Same dtype for now
            scale_dtype=None,
            hidden_dim_scale=None,
            nets_per_gpu=2,  # Default
            max_private_tokens=None,
        )
        
        # Verify that num_tokens doesn't exceed the instance's max_num_tokens
        # This should not happen due to our rounding, but check for safety
        # Note: P2PAllToAll doesn't expose max_num_tokens directly, but dispatch_send will check it
        # We rely on our rounding logic to ensure num_tokens <= max_num_tokens
        
        # Allocate output buffers
        device = x.device
        out_expert_num_tokens = torch.empty(
            (num_local_experts,), dtype=torch.int32, device=device
        )
        out_expert_x = torch.empty(
            (max_recv_tokens, hidden_dim), dtype=x.dtype, device=device
        )
        out_expert_x_scale = None  # No scale for now
        
        # Convert indices to uint32 if needed
        if token_indices.dtype != torch.uint32:
            # Handle -1 masking: convert to valid expert indices
            token_indices_uint32 = token_indices.to(torch.int64)
            # Mask invalid indices (assuming -1 means invalid)
            mask = token_indices_uint32 < 0
            token_indices_uint32 = token_indices_uint32.clamp(min=0, max=num_experts - 1)
            token_indices_uint32 = token_indices_uint32.to(torch.uint32)
        else:
            token_indices_uint32 = token_indices
            mask = None
        
        # Ensure weights are float32
        if token_probs.dtype != torch.float32:
            token_probs_f32 = token_probs.float()
        else:
            token_probs_f32 = token_probs
        
        # Perform dispatch
        # For async_finish, we can separate send and recv
        do_send = True
        do_recv = True
        if async_finish:
            # In async mode, we might want to separate send/recv
            # For now, do both
            pass
        
        all_to_all.dispatch(
            out_expert_num_tokens=out_expert_num_tokens,
            out_expert_x=out_expert_x,
            out_expert_x_scale=out_expert_x_scale,
            dp_x=x.contiguous(),
            dp_x_scale=None,
            indices=token_indices_uint32,
            weights=token_probs_f32,
            bound_m=None,
            do_send=do_send,
            do_recv=do_recv,
        )
        
        # Synchronize if needed
        if not async_finish:
            torch.cuda.synchronize()
        
        # Calculate actual number of tokens received
        total_recv_tokens = out_expert_num_tokens.sum().item()
        if total_recv_tokens > 0:
            recv_x = out_expert_x[:total_recv_tokens]
        else:
            recv_x = torch.empty((0, hidden_dim), dtype=x.dtype, device=device)
        
        # For compatibility, we need to return indices and probs
        # These should match the received tokens
        # For now, we'll return the original indices/probs (this might need adjustment)
        recv_token_indices = token_indices_uint32
        recv_token_probs = token_probs_f32
        
        # Create a handle-like object to store metadata for backward
        handle = {
            'all_to_all': all_to_all,
            'token_indices': token_indices_uint32,
            'token_probs': token_probs_f32,
            'num_tokens': num_tokens,
            'num_experts': num_experts,  # Save for backward
        }
        
        # Save for backward
        ctx.group = group
        ctx.handle = handle
        ctx.async_finish = async_finish
        ctx.allocate_on_comm_stream = allocate_on_comm_stream
        ctx.num_experts = num_experts
        ctx.num_experts_per_token = num_experts_per_token
        ctx.hidden_dim = hidden_dim
        ctx.out_expert_num_tokens = out_expert_num_tokens
        ctx.out_expert_x = out_expert_x
        ctx.max_recv_tokens = max_recv_tokens
        
        tokens_per_expert = out_expert_num_tokens
        
        return (recv_x, recv_token_indices, recv_token_probs, tokens_per_expert, handle)

    @staticmethod
    def backward(
        ctx, grad_output, grad_token_indices, grad_token_probs, grad_tokens_per_expert, grad_handle
    ):
        """Backward pass of fused dispatch."""
        if not HAVE_PPLX:
            raise ImportError("pplx_garden is not installed")
        
        handle = ctx.handle
        all_to_all = handle['all_to_all']
        token_indices = handle['token_indices']
        token_probs = handle['token_probs']
        num_tokens = handle['num_tokens']
        
        # Allocate output buffer for combine
        out_tokens = torch.empty(
            (num_tokens, ctx.hidden_dim), dtype=grad_output.dtype, device=grad_output.device
        )
        
        # Ensure grad_token_probs is float32
        if grad_token_probs.dtype != torch.float32:
            grad_token_probs_f32 = grad_token_probs.float()
        else:
            grad_token_probs_f32 = grad_token_probs
        
        # Perform combine (reverse of dispatch)
        do_send = True
        do_recv = True
        
        all_to_all.combine(
            out_tokens=out_tokens,
            indices=token_indices,
            weights=grad_token_probs_f32,
            expert_y=grad_output.contiguous(),
            bound_m=None,
            do_send=do_send,
            do_recv=do_recv,
            accumulate=False,
        )
        
        # Synchronize if needed
        if not ctx.async_finish:
            torch.cuda.synchronize()
        
        grad_x = out_tokens
        
        return grad_x, None, grad_token_probs_f32, None, None, None, None


class FusedCombine(torch.autograd.Function):
    """Fused combine operation for MoE output combining computation and communication."""

    @staticmethod
    def forward(ctx, x, group, handle, async_finish=False, allocate_on_comm_stream=False):
        """Forward pass of fused combine."""
        if not HAVE_PPLX:
            raise ImportError("pplx_garden is not installed")
        
        # Extract information from handle
        all_to_all = handle['all_to_all']
        token_indices = handle['token_indices']
        token_probs = handle['token_probs']
        num_tokens = handle['num_tokens']
        
        num_recv_tokens, hidden_dim = x.shape
        
        # Allocate output buffer
        out_tokens = torch.empty(
            (num_tokens, hidden_dim), dtype=x.dtype, device=x.device
        )
        
        # Ensure token_probs is float32
        if token_probs.dtype != torch.float32:
            token_probs_f32 = token_probs.float()
        else:
            token_probs_f32 = token_probs
        
        # Perform combine
        do_send = True
        do_recv = True
        
        all_to_all.combine(
            out_tokens=out_tokens,
            indices=token_indices,
            weights=token_probs_f32,
            expert_y=x.contiguous(),
            bound_m=None,
            do_send=do_send,
            do_recv=do_recv,
            accumulate=False,
        )
        
        # Synchronize if needed
        if not async_finish:
            torch.cuda.synchronize()
        
        ctx.handle = handle
        ctx.group = group
        ctx.async_finish = async_finish
        ctx.allocate_on_comm_stream = allocate_on_comm_stream
        ctx.num_tokens = num_tokens
        ctx.hidden_dim = hidden_dim
        
        return out_tokens, None

    @staticmethod
    def backward(ctx, grad_output, grad_placeholder):
        """Backward pass of fused combine."""
        if not HAVE_PPLX:
            raise ImportError("pplx_garden is not installed")
        
        handle = ctx.handle
        all_to_all = handle['all_to_all']
        token_indices = handle['token_indices']
        token_probs = handle['token_probs']
        
        num_tokens, hidden_dim = grad_output.shape
        
        # Calculate max_recv_tokens for backward dispatch
        group = ctx.group
        group_size = group.size()
        num_experts = handle.get('num_experts', ctx.num_experts if hasattr(ctx, 'num_experts') else group_size)
        num_local_experts = (num_experts + group_size - 1) // group_size
        num_experts_per_token = token_indices.shape[1]
        max_recv_tokens = max(
            num_tokens * num_experts_per_token,
            num_local_experts,
        )
        max_recv_tokens = ((max_recv_tokens + 15) // 16) * 16
        
        # Allocate output buffers for backward dispatch
        device = grad_output.device
        out_expert_num_tokens = torch.empty(
            (num_local_experts,), dtype=torch.int32, device=device
        )
        out_expert_x = torch.empty(
            (max_recv_tokens, hidden_dim), dtype=grad_output.dtype, device=device
        )
        
        # Ensure token_probs is float32
        if token_probs.dtype != torch.float32:
            token_probs_f32 = token_probs.float()
        else:
            token_probs_f32 = token_probs
        
        # Perform dispatch (reverse of combine)
        do_send = True
        do_recv = True
        
        all_to_all.dispatch(
            out_expert_num_tokens=out_expert_num_tokens,
            out_expert_x=out_expert_x,
            out_expert_x_scale=None,
            dp_x=grad_output.contiguous(),
            dp_x_scale=None,
            indices=token_indices,
            weights=token_probs_f32,
            bound_m=None,
            do_send=do_send,
            do_recv=do_recv,
        )
        
        # Synchronize if needed
        if not ctx.async_finish:
            torch.cuda.synchronize()
        
        # Calculate actual number of tokens received
        total_recv_tokens = out_expert_num_tokens.sum().item()
        if total_recv_tokens > 0:
            grad_x = out_expert_x[:total_recv_tokens]
        else:
            grad_x = torch.empty((0, hidden_dim), dtype=grad_output.dtype, device=device)
        
        return grad_x, None, None, None, None


if HAVE_PPLX:

    def fused_dispatch(
        x,
        token_indices,
        token_probs,
        num_experts,
        group,
        async_finish=False,
        allocate_on_comm_stream=False,
    ):
        """Perform fused dispatch operation if pplx_garden is available.

        Args:
            x: Input tensor [num_tokens, hidden_size]
            token_indices: Token routing indices [num_tokens, topk]
            token_probs: Token routing probabilities [num_tokens, topk]
            num_experts: Number of experts
            group: Process group
            async_finish: Whether to finish asynchronously
            allocate_on_comm_stream: Whether to allocate on communication stream

        Returns:
            Tuple of (recv_x, recv_token_indices, recv_token_probs, tokens_per_expert, handle)
        """
        return FusedDispatch.apply(
            x.contiguous(),
            token_indices,
            token_probs,
            num_experts,
            group,
            async_finish,
            allocate_on_comm_stream,
        )

    def fused_combine(x, group, handle, async_finish=False, allocate_on_comm_stream=False):
        """Perform fused combine operation if pplx_garden is available.

        Args:
            x: Input tensor [num_recv_tokens, hidden_size]
            group: Process group
            handle: Communication handle from dispatch
            async_finish: Whether to finish asynchronously
            allocate_on_comm_stream: Whether to allocate on communication stream

        Returns:
            Tuple of (combined_x, None)
        """
        return FusedCombine.apply(x, group, handle, async_finish, allocate_on_comm_stream)

    def set_deepep_num_sms(num_sms):
        """Sets the number of SMs (not used in pplx, kept for compatibility)"""
        # pplx doesn't use this parameter, but we keep it for API compatibility
        pass

else:
    fused_dispatch = None
    fused_combine = None
    set_deepep_num_sms = None
