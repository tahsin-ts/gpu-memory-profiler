"""A simulator of PyTorch's CUDA caching allocator (the default, without expandable segments).

PyTorch does not return freed memory to CUDA. It keeps freed blocks in pools and reuses them,
splitting big blocks for smaller requests. That is fast, but it can FRAGMENT memory: plenty is
"reserved but unallocated", yet no single free block is big enough, and the job runs out of memory.

The rules below follow c10/cuda/CUDACachingAllocator.cpp:
  * request sizes are rounded up to a multiple of 512 bytes
  * requests <= 1 MiB use the SMALL pool, carved from 2 MiB segments
  * larger requests use the LARGE pool: < 10 MiB get a 20 MiB segment, others are rounded to 2 MiB
  * best fit: the smallest cached free block that is big enough
  * a block is split if the rest is >= 512 B (small pool) or > 1 MiB (large pool)
  * freed blocks merge with free neighbours in the same segment
  * if CUDA has no room, all fully free cached segments are released and the request retried
  * max_split_size: blocks at least this big are never split, and are not used for smaller requests
It is a model for learning and planning, not a byte-exact copy of every PyTorch version.
"""
import bisect
import itertools

from .units import MiB, fmt

MIN_BLOCK = 512
SMALL_SIZE = 1 * MiB
SMALL_BUFFER = 2 * MiB
LARGE_BUFFER = 20 * MiB
MIN_LARGE_ALLOC = 10 * MiB
ROUND_LARGE = 2 * MiB


class OutOfMemory(RuntimeError):
    pass


def round_size(n: int) -> int:
    return MIN_BLOCK if n < MIN_BLOCK else MIN_BLOCK * ((n + MIN_BLOCK - 1) // MIN_BLOCK)


def segment_size(n: int) -> int:
    if n <= SMALL_SIZE:
        return SMALL_BUFFER
    if n < MIN_LARGE_ALLOC:
        return LARGE_BUFFER
    return ROUND_LARGE * ((n + ROUND_LARGE - 1) // ROUND_LARGE)


class Block:
    __slots__ = ("addr", "size", "requested", "allocated", "segment", "prev", "next", "tag")

    def __init__(self, addr, size, segment):
        self.addr, self.size, self.segment = addr, size, segment
        self.requested, self.allocated, self.prev, self.next, self.tag = 0, False, None, None, None


class Segment:
    __slots__ = ("addr", "size", "pool", "head")

    def __init__(self, addr, size, pool):
        self.addr, self.size, self.pool = addr, size, pool
        self.head = Block(addr, size, self)


class CachingAllocator:
    def __init__(self, capacity: int, max_split_size: int = None):
        self.capacity = capacity
        self.max_split = max_split_size if max_split_size else float("inf")
        self.free_lists = {"small": [], "large": []}      # sorted (size, addr) keys
        self.free_blocks = {}                              # addr -> free Block
        self.live = {}                                     # addr -> allocated Block
        self.segments = {}                                 # addr -> Segment
        self.next_addr = 0x7F0000000000
        self.reserved = self.allocated = self.requested = 0
        self.peak_allocated = self.peak_reserved = 0
        self.num_alloc_retries = self.num_ooms = self.num_segment_allocs = 0
        self.trace = []                                    # events in torch snapshot style
        self._clock = itertools.count()

    # ---------------------------------------------------------------- free-list helpers
    def _push_free(self, b):
        bisect.insort(self.free_lists[b.segment.pool], (b.size, b.addr))
        self.free_blocks[b.addr] = b

    def _pop_free(self, b):
        lst = self.free_lists[b.segment.pool]
        i = bisect.bisect_left(lst, (b.size, b.addr))
        assert lst[i] == (b.size, b.addr)
        del lst[i]
        del self.free_blocks[b.addr]

    def _find(self, pool, size):
        lst = self.free_lists[pool]
        i = bisect.bisect_left(lst, (size, 0))
        if i == len(lst):
            return None
        bsize, addr = lst[i]
        if pool == "large":
            if size < self.max_split and bsize >= self.max_split:
                return None                                # do not cut an oversize block for a small request
            if size >= self.max_split and bsize >= size + LARGE_BUFFER:
                return None                                # an oversize request must not waste a much bigger block
        return self.free_blocks[addr]

    def _should_split(self, b, size):
        rest = b.size - size
        if b.segment.pool == "small":
            return rest >= MIN_BLOCK
        return size < self.max_split and rest > SMALL_SIZE

    def _event(self, action, addr, size, tag=None):
        ev = {"action": action, "addr": addr, "size": size, "stream": 0, "time_us": next(self._clock)}
        if tag:
            ev["frames"] = [{"filename": tag, "line": 0, "name": tag}]
        self.trace.append(ev)

    # ---------------------------------------------------------------- public API
    def malloc(self, nbytes: int, tag: str = None) -> int:
        size = round_size(nbytes)
        pool = "small" if size <= SMALL_SIZE else "large"
        b = self._find(pool, size)
        if b is not None:
            self._pop_free(b)
        else:
            seg_bytes = segment_size(size)
            if self.reserved + seg_bytes > self.capacity:
                self.num_alloc_retries += 1
                self.empty_cache()                         # give fully free segments back, then retry
                if self.reserved + seg_bytes > self.capacity:
                    self.num_ooms += 1
                    self._event("oom", 0, size, tag)
                    raise OutOfMemory(self.oom_message(size))
            seg = Segment(self.next_addr, seg_bytes, pool)
            self.next_addr += seg_bytes
            self.segments[seg.addr] = seg
            self.reserved += seg_bytes
            self.num_segment_allocs += 1
            self._event("segment_alloc", seg.addr, seg_bytes)
            b = seg.head
        if self._should_split(b, size):
            rest = Block(b.addr + size, b.size - size, b.segment)
            rest.prev, rest.next = b, b.next
            if b.next:
                b.next.prev = rest
            b.next, b.size = rest, size
            self._push_free(rest)
        b.allocated, b.requested, b.tag = True, nbytes, tag
        self.live[b.addr] = b
        self.allocated += b.size
        self.requested += nbytes
        self.peak_allocated = max(self.peak_allocated, self.allocated)
        self.peak_reserved = max(self.peak_reserved, self.reserved)
        self._event("alloc", b.addr, b.size, tag)
        return b.addr

    def free(self, addr: int) -> None:
        b = self.live.pop(addr)
        self._event("free_completed", b.addr, b.size, b.tag)
        self.allocated -= b.size
        self.requested -= b.requested
        b.allocated, b.requested, b.tag = False, 0, None
        for nb in (b.prev, b.next):                        # merge with free neighbours
            if nb is not None and not nb.allocated:
                self._pop_free(nb)
                first, second = (nb, b) if nb is b.prev else (b, nb)
                first.size += second.size
                first.next = second.next
                if second.next:
                    second.next.prev = first
                b = first
        self._push_free(b)

    def empty_cache(self) -> int:
        """Release every segment that is completely free (one free block covering it)."""
        released = 0
        for addr, seg in list(self.segments.items()):
            h = seg.head
            if not h.allocated and h.next is None and h.size == seg.size:
                self._pop_free(h)
                del self.segments[addr]
                self.reserved -= seg.size
                released += seg.size
                self._event("segment_free", addr, seg.size)
        return released

    # ---------------------------------------------------------------- reporting
    def largest_free_block(self) -> int:
        sizes = [s for lst in self.free_lists.values() for s, _ in lst[-1:]]
        return max(sizes, default=0)

    def stats(self) -> dict:
        return {"allocated": self.allocated, "reserved": self.reserved, "requested": self.requested,
                "peak_allocated": self.peak_allocated, "peak_reserved": self.peak_reserved,
                "fragmentation": 1 - self.allocated / self.reserved if self.reserved else 0.0,
                "largest_free_block": self.largest_free_block(), "segments": len(self.segments),
                "num_alloc_retries": self.num_alloc_retries, "num_ooms": self.num_ooms}

    def oom_message(self, size: int) -> str:
        return (f"CUDA out of memory. Tried to allocate {fmt(size)}. GPU 0 has a total capacity of {fmt(self.capacity)} "
                f"of which {fmt(self.capacity - self.reserved)} is free. Of the allocated memory {fmt(self.allocated)} "
                f"is allocated by PyTorch, and {fmt(self.reserved - self.allocated)} is reserved by PyTorch but unallocated. "
                f"Largest cached free block: {fmt(self.largest_free_block())}. If reserved but unallocated memory is large, "
                f"fragmentation is the cause: try PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True or max_split_size_mb.")

    def snapshot(self) -> dict:
        """Same shape as torch.cuda.memory._snapshot(), so gpumem.snapshot can analyse both."""
        segs = []
        for seg in sorted(self.segments.values(), key=lambda s: s.addr):
            blocks, b = [], seg.head
            while b:
                blk = {"address": b.addr, "size": b.size, "requested_size": b.requested,
                       "state": "active_allocated" if b.allocated else "inactive"}
                if b.tag:
                    blk["frames"] = [{"filename": b.tag, "line": 0, "name": b.tag}]
                blocks.append(blk)
                b = b.next
            alloc = sum(x["size"] for x in blocks if x["state"] == "active_allocated")
            segs.append({"device": 0, "address": seg.addr, "total_size": seg.size, "allocated_size": alloc,
                         "active_size": alloc, "stream": 0, "segment_type": seg.pool, "blocks": blocks})
        return {"segments": segs, "device_traces": [list(self.trace)]}
