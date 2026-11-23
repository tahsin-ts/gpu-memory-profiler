"""Synthetic workloads that reproduce common GPU-memory failures in the simulator."""
import random

from .allocator import CachingAllocator, OutOfMemory
from .units import MiB


def varlen_training(alloc: CachingAllocator, steps=300, seed=0, layers=24, hidden_bytes=8 * 1024,
                    min_seq=128, max_seq=4096, keep_every=1, keep_bytes=6 * MiB, keep_window=40):
    """A training loop whose batches have different sequence lengths.

    Each step allocates per-layer activations sized by the step's sequence length and frees them at
    the end of the step. Meanwhile some small tensors outlive the step (e.g. metrics or gradient
    buckets kept for a while). Those long-lived tensors get carved out of the big cached activation
    blocks and pin them, which is the classic way fragmentation builds up.
    Returns (steps_completed, error_message_or_None).
    """
    rnd = random.Random(seed)
    alloc.malloc(1536 * MiB, "weights")                       # persistent state
    kept = []
    for step in range(steps):
        seq = rnd.randint(min_seq, max_seq)
        acts = []
        try:
            for layer in range(layers):
                acts.append(alloc.malloc(seq * hidden_bytes * 4, f"activation L{layer}"))
                if layer == layers // 2 and step % keep_every == 0:
                    kept.append(alloc.malloc(keep_bytes, "kept tensor"))
                    if len(kept) > keep_window:
                        alloc.free(kept.pop(0))
        except OutOfMemory as e:
            return step, str(e)
        for a in acts:
            alloc.free(a)
    return steps, None
