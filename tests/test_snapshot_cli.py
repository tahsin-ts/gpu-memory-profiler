import pickle

from gpumem.allocator import CachingAllocator
from gpumem.cli import main
from gpumem.snapshot import summarize, timeline
from gpumem.units import MiB


def test_summary_matches_allocator_state():
    a = CachingAllocator(1 << 30)
    keep = [a.malloc(3 * MiB, "model.py:10 forward")]
    tmp = [a.malloc(5 * MiB, "loss.py:3 loss") for _ in range(3)]
    for t in tmp:
        a.free(t)
    s = summarize(a.snapshot())
    assert s["reserved"] == a.reserved and s["allocated"] == a.allocated
    assert s["top_sites"][0][0] == "model.py:10 forward"
    t = timeline(a.snapshot())
    assert t["peak"] == a.peak_allocated
    assert dict(t["at_peak"])["loss.py:3 loss"] == 15 * MiB


def test_cli_estimate_simulate_analyze(tmp_path, capsys):
    assert main(["estimate", "--model", "llama3-8b", "--mode", "infer", "--batch", "4", "--seq", "8192"]) == 0
    assert "KV cache per token: 128.00 KiB" in capsys.readouterr().out
    snap = tmp_path / "s.pkl"
    assert main(["simulate", "--capacity-gib", "6", "--snapshot", str(snap)]) == 1      # OOM expected
    assert "reserved by PyTorch but unallocated" in capsys.readouterr().out
    assert main(["analyze", str(snap)]) == 0
    out = capsys.readouterr().out
    assert "pinned by small live blocks" in out and "OOM at event" in out
    assert pickle.load(open(snap, "rb"))["segments"]
