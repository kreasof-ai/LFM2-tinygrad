# /paged_attention.py (Corrected)

"""
Paged Attention implementation for tinygrad.

This module provides classes for managing a paged key-value (KV) cache, 
allowing for efficient memory management during batched inference of large
language models. It is inspired by the vLLM project and the attention-gym
reference implementation.

The core idea is to break the KV cache into non-contiguous blocks (pages)
and use a `PageTable` to map the logical sequence of tokens for each request
to the physical pages in memory.

Since tinygrad does not have a `flex_attention` equivalent, this implementation
uses `gather` to read from the cache and `scatter` to write to it.
"""
from __future__ import annotations
from typing import List, Tuple

from tinygrad import Tensor, dtypes, Device

def _cdiv(x: int, multiple: int) -> int:
    return (x + multiple - 1) // multiple

class PageTable:
    """
    Manages the mapping from logical blocks to physical pages for a batch of sequences.
    
    This class handles the allocation, reservation, and freeing of pages and provides
    the logic to calculate physical addresses for scatter/gather operations.
    """
    def __init__(self, n_pages: int, page_size: int, max_batch_size: int):
        self.n_pages = n_pages
        self.page_size = page_size
        self.max_batch_size = max_batch_size
        
        # A reasonable starting max seq len. The table can grow.
        initial_blocks = _cdiv(4096, page_size) 
        self.page_table = Tensor.full((max_batch_size, initial_blocks), -1, dtype=dtypes.int32).realize()
        self.page_table_cpu = [[-1] * initial_blocks for _ in range(max_batch_size)]

        self.capacity = [0] * max_batch_size  # capacity in tokens
        self.free_pages = list(reversed(range(n_pages)))
        self.free_batch_idx = list(reversed(range(max_batch_size)))

    def _ensure_table_capacity(self, num_logical_blocks: int):
        """Dynamically expand the page table if needed."""
        if num_logical_blocks > self.page_table.shape[1]:
            new_cols = num_logical_blocks - self.page_table.shape[1]
            padding_tensor = Tensor.full((self.max_batch_size, new_cols), -1, dtype=dtypes.int32)
            self.page_table = self.page_table.cat(padding_tensor, dim=1).realize()
            for i in range(self.max_batch_size):
                self.page_table_cpu[i].extend([-1] * new_cols)

    def allocate(self) -> int:
        """Allocates a new slot in the batch. Returns the batch index."""
        if not self.free_batch_idx:
            raise RuntimeError("Cannot allocate new batch slot, max_batch_size reached.")
        batch_idx = self.free_batch_idx.pop()
        self.capacity[batch_idx] = 0
        return batch_idx

    def reserve(self, batch_idx: int, seq_len: int) -> bool:
        """Ensures a batch item has enough capacity for a given sequence length."""
        if seq_len <= self.capacity[batch_idx]:
            return True

        num_logical_blocks_needed = _cdiv(seq_len, self.page_size)
        self._ensure_table_capacity(num_logical_blocks_needed)
        
        current_pages = _cdiv(self.capacity[batch_idx], self.page_size)
        pages_to_allocate = num_logical_blocks_needed - current_pages

        if pages_to_allocate > len(self.free_pages):
            print(f"Warning: Failed to reserve {pages_to_allocate} pages for batch {batch_idx}. Only {len(self.free_pages)} available.")
            return False

        newly_allocated_pages = [self.free_pages.pop() for _ in range(pages_to_allocate)]
        
        # *** FIX START ***
        # To update specific (row, col) coordinates, we flatten the tensor, calculate
        # the 1D indices, and then scatter along dim=0 of the flattened tensor.
        num_cols = self.page_table.shape[1]
        flat_indices = Tensor([batch_idx * num_cols + i for i in range(current_pages, num_logical_blocks_needed)], dtype=dtypes.int32)
        src = Tensor(newly_allocated_pages, dtype=dtypes.int32)

        if flat_indices.shape[0] > 0: # only scatter if there are indices to update
            self.page_table = self.page_table.flatten().scatter(0, flat_indices, src).reshape(self.page_table.shape).realize()

        # Update the CPU copy
        for i, page_idx in enumerate(newly_allocated_pages):
            self.page_table_cpu[batch_idx][current_pages + i] = page_idx
        # *** FIX END ***

        self.capacity[batch_idx] = num_logical_blocks_needed * self.page_size
        return True

    def erase(self, batch_idx: int):
        """Frees all pages associated with a batch index."""
        num_logical_blocks = _cdiv(self.capacity[batch_idx], self.page_size)
        for i in range(num_logical_blocks):
            page = self.page_table_cpu[batch_idx][i]
            if page != -1:
                self.free_pages.append(page)
        
        # *** FIX START ***
        # Use the same flatten-scatter-reshape pattern to reset values to -1.
        num_cols = self.page_table.shape[1]
        flat_indices = Tensor([batch_idx * num_cols + i for i in range(num_logical_blocks)], dtype=dtypes.int32)
        src = Tensor.full((num_logical_blocks,), -1, dtype=dtypes.int32)

        if flat_indices.shape[0] > 0:
            self.page_table = self.page_table.flatten().scatter(0, flat_indices, src).reshape(self.page_table.shape).realize()

        # Reset the CPU copy
        for i in range(num_logical_blocks):
            self.page_table_cpu[batch_idx][i] = -1
        # *** FIX END ***

        self.free_batch_idx.append(batch_idx)
        self.capacity[batch_idx] = 0

    def get_physical_addrs(self, batch_indices: Tensor, positions: Tensor) -> Tensor:
        """
        Calculates the physical addresses in the flat cache tensor.
        This version operates entirely on the device (GPU).
        """
        logical_block_idx = (positions // self.page_size).cast(dtypes.int32)
        logical_block_offset = (positions % self.page_size).cast(dtypes.int32)

        # Flatten the page table to easily index into it.
        # Shape: (max_batch_size * max_logical_blocks)
        page_table_flat = self.page_table.flatten()
        num_logical_blocks_per_row = self.page_table.shape[1]

        # Calculate the flat index for each token's logical block
        # flat_index = row * num_cols + col
        flat_indices = batch_indices * num_logical_blocks_per_row + logical_block_idx
        
        # Gather the physical block indices from the flattened table
        physical_block_idx = page_table_flat.gather(0, flat_indices.flatten()).reshape(positions.shape)
        
        return (physical_block_idx * self.page_size + logical_block_offset).cast(dtypes.int32)

# The rest of the file (PagedKVCache) remains unchanged as it uses get_physical_addrs,
# which we have now corrected. I'll include it for completeness.

class PagedKVCache:
    """
    A stateful container for the physical K and V caches and their corresponding PageTable.
    """
    def __init__(self, page_table: PageTable, num_heads: int, head_dim: int, dtype):
        super().__init__()
        self.page_table = page_table
        self.num_heads = num_heads
        self.head_dim = head_dim
        
        cache_shape = (1, num_heads, page_table.n_pages * page_table.page_size, head_dim)
        self.k_cache = Tensor.zeros(cache_shape, dtype=dtype, device=Device.DEFAULT)
        self.v_cache = Tensor.zeros(cache_shape, dtype=dtype, device=Device.DEFAULT)

    def update(self, batch_idx: Tensor, input_pos: Tensor, k_val: Tensor, v_val: Tensor):
        """
        Writes new key-value pairs into the physical cache using scatter.
        Args:
            batch_idx (Tensor): Shape (B,). The batch index for each sequence.
            input_pos (Tensor): Shape (B, S). The logical positions of new tokens.
            k_val (Tensor): Shape (B, S, num_heads, head_dim). New key values.
            v_val (Tensor): Shape (B, S, num_heads, head_dim). New value values.
        """
        B, S, H, D = k_val.shape
        assert H == self.num_heads and D == self.head_dim
        assert batch_idx.shape == (B,) and input_pos.shape == (B, S)

        k_val = k_val.permute(0, 2, 1, 3).reshape(1, H, B * S, D)
        v_val = v_val.permute(0, 2, 1, 3).reshape(1, H, B * S, D)

        batch_indices = batch_idx.reshape(B, 1).expand(B, S)
        
        addrs = self.page_table.get_physical_addrs(batch_indices, input_pos).flatten()
        
        scatter_indices = addrs.reshape(1, 1, -1, 1).expand(1, H, B * S, D)

        # *** FIX START ***
        # Correct the argument order: scatter(dim, index, src)
        self.k_cache = self.k_cache.scatter(2, scatter_indices, k_val).realize()
        self.v_cache = self.v_cache.scatter(2, scatter_indices, v_val).realize()
        # *** FIX END ***

    # In /paged_attention.py -> class PagedKVCache

    def gather_kv_for_attention(self, batch_idx: Tensor, seq_lens: List[int]) -> Tuple[Tensor, Tensor]:
        """
        Gathers all required K/V pairs for an attention computation.
        Args:
            batch_idx (Tensor): Shape (B,). The batch indices for the current attention op.
            seq_lens (List[int]): A list of logical sequence lengths for each item in the batch.
        Returns:
            Tuple[Tensor, Tensor]: Padded K and V tensors of shape (B, max_seq_len, H, D).
        """
        B = batch_idx.shape[0]
        max_seq_len = max(seq_lens)
        
        positions = Tensor.arange(max_seq_len).reshape(1, max_seq_len).expand(B, max_seq_len)
        batch_indices = batch_idx.reshape(B, 1).expand(B, max_seq_len)
        addrs = self.page_table.get_physical_addrs(batch_indices, positions).flatten()
        gather_indices = addrs.reshape(1, 1, -1, 1).expand(1, self.num_heads, -1, self.head_dim)
        
        # *** FIX START ***
        # Correct the argument order: gather(dim, index)
        gathered_k = self.k_cache.gather(2, gather_indices).reshape(B, self.num_heads, max_seq_len, self.head_dim).permute(0, 2, 1, 3)
        gathered_v = self.v_cache.gather(2, gather_indices).reshape(B, self.num_heads, max_seq_len, self.head_dim).permute(0, 2, 1, 3)
        # *** FIX END ***

        return gathered_k, gathered_v