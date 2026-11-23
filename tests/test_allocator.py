import pytest

from gpumem.allocator import (LARGE_BUFFER, CachingAllocator, OutOfMemory, round_size, segment_size)
from gpumem.units import GiB, MiB
from gpumem.workloads import varlen_training


def test_rounding_and_segment_sizes():
    assert round_size(1) == 512 and round_size(512) == 512 and round_size(513) == 1024
    assert segment_size(4096) == 2 * MiB                 # small pool
    assert segment_size(5 * MiB) == 20 * MiB             # medium: a 20 MiB segment
    assert segment_size(11 * MiB + 1) == 12 * MiB        # large: rounded up to 2 MiB


def test_small_requests_share_one_segment_and_reuse_memory():
    a = CachingAllocator(1 * GiB)
    x = [a.malloc(1000) for _ in range(100)]
    assert a.stats()["segments"] == 1 and a.reserved == 2 * MiB
    for p in x:
        a.free(p)
    assert a.allocated == 0 and a.reserved == 2 * MiB      # cached, not returned to CUDA
    a.malloc(1000)
    assert a.num_segment_allocs == 1                       # reused


def test_split_and_merge():
    a = CachingAllocator(1 * GiB)
    p1 = a.malloc(4 * MiB)                                 # 20 MiB segment, 16 MiB left free
    p2 = a.malloc(4 * MiB)
    assert a.stats()["segments"] == 1
    a.free(p1)
    a.free(p2)
    assert a.largest_free_block() == LARGE_BUFFER          # all three pieces merged back
    assert a.empty_cache() == LARGE_BUFFER and a.reserved == 0


def test_large_block_not_split_when_rest_is_small():
    a = CachingAllocator(1 * GiB)
    p = a.malloc(12 * MiB)
    a.free(p)
    q = a.malloc(11 * MiB + 512)                           # rest 512 KiB <= 1 MiB: whole block is used
    assert a.live[q].size == 12 * MiB


def test_retry_releases_cache_before_oom():
    a = CachingAllocator(100 * MiB)
    p = a.malloc(60 * MiB)
    a.free(p)                                              # 60 MiB cached and unused
    a.malloc(70 * MiB)                                     # cannot fit next to it: cache released, retry succeeds
    assert a.num_alloc_retries == 1 and a.num_ooms == 0


def test_oom_message_reports_fragmentation():
    a = CachingAllocator(100 * MiB)
    a.malloc(90 * MiB)
    with pytest.raises(OutOfMemory) as e:
        a.malloc(20 * MiB)
    assert "Tried to allocate 20.00 MiB" in str(e.value) and a.num_ooms == 1


def test_max_split_size_prevents_the_fragmentation_oom():
    plain = CachingAllocator(6 * GiB)
    steps, err = varlen_training(plain)
    assert err is not None and steps < 300                  # fragmentation OOM, although peak use < 5 GiB
    assert plain.peak_allocated < 5 * GiB
    tuned = CachingAllocator(6 * GiB, max_split_size=128 * MiB)
    steps, err = varlen_training(tuned)
    assert err is None and steps == 300


def test_invariants_hold_under_random_traffic():
    import random
    r = random.Random(1)
    a = CachingAllocator(4 * GiB)
    live = []
    for _ in range(3000):
        if live and r.random() < 0.45:
            a.free(live.pop(r.randrange(len(live))))
        else:
            live.append(a.malloc(r.choice([100, 5000, 600_000, 3 * MiB, 30 * MiB])))
        if r.random() < 0.01:
            a.empty_cache()
    for seg in a.segments.values():                        # blocks tile each segment exactly, no free neighbours
        b, total = seg.head, 0
        while b:
            total += b.size
            assert not (b.next and not b.allocated and not b.next.allocated)
            b = b.next
        assert total == seg.size
    assert a.allocated == sum(a.live[p].size for p in live)
