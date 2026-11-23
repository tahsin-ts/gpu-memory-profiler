"""Helpers for real PyTorch jobs (need torch and a CUDA GPU)."""
import contextlib

import torch


@contextlib.contextmanager
def record(path: str, max_entries: int = 100_000):
    """Record every allocation with its Python stack, then write a snapshot for `gpumem analyze`.
    Also viewable at https://pytorch.org/memory_viz."""
    torch.cuda.memory._record_memory_history(max_entries=max_entries)
    try:
        yield
    finally:
        torch.cuda.memory._dump_snapshot(path)
        torch.cuda.memory._record_memory_history(enabled=None)


def step_stats(device: int = 0) -> dict:
    """The few numbers worth logging every N steps."""
    s = torch.cuda.memory_stats(device)
    return {"allocated": s["allocated_bytes.all.current"], "reserved": s["reserved_bytes.all.current"],
            "peak_allocated": s["allocated_bytes.all.peak"], "alloc_retries": s["num_alloc_retries"],
            "ooms": s["num_ooms"]}
