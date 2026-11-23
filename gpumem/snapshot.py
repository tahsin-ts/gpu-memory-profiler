"""Analyse a memory snapshot: from torch.cuda.memory._snapshot() / _dump_snapshot(), or from the simulator.

Snapshot shape used (PyTorch 2.x; unknown fields are ignored):
  {"segments": [{"address", "total_size", "segment_type", "blocks": [{"address", "size", "state", "frames"}]}],
   "device_traces": [[{"action": "alloc" | "free_completed" | "segment_alloc" | "segment_free" | "oom", "addr", "size", "frames"}]]}
"""
import collections
import pickle

from .units import fmt


def load(path: str) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


def site(frames) -> str:
    """Where an allocation came from: the first frame outside PyTorch internals."""
    for fr in frames or []:
        fn = fr.get("filename", "")
        if "torch/" not in fn and "site-packages" not in fn and fn != "??":
            name = fr.get("name", "")
            return name if name == fn else f"{fn}:{fr.get('line', 0)} {name}".strip()
    return "unknown"


def summarize(snap: dict, top: int = 5) -> dict:
    segs = snap.get("segments", [])
    reserved = sum(s["total_size"] for s in segs)
    active = [b for s in segs for b in s["blocks"] if b["state"].startswith("active")]
    free = [b for s in segs for b in s["blocks"] if b["state"] == "inactive"]
    allocated = sum(b["size"] for b in active)
    by_site = collections.Counter()
    for b in active:
        by_site[site(b.get("frames"))] += b["size"]
    worst = sorted(((s["total_size"] - sum(b["size"] for b in s["blocks"] if b["state"].startswith("active")), s["total_size"],
                     s["address"]) for s in segs), reverse=True)[:top]
    return {"reserved": reserved, "allocated": allocated, "unallocated": reserved - allocated,
            "fragmentation": (1 - allocated / reserved) if reserved else 0.0,
            "largest_free_block": max((b["size"] for b in free), default=0),
            "segments": len(segs), "free_blocks": len(free),
            "top_sites": by_site.most_common(top),
            "most_wasteful_segments": [(addr, total, waste) for waste, total, addr in worst if waste > 0]}


def timeline(snap: dict, device: int = 0):
    """Replay the trace: allocated bytes after every event, the peak, and what was alive at the peak."""
    alive, cur, peak, peak_i, peak_alive, series, ooms = {}, 0, 0, -1, {}, [], []
    for i, ev in enumerate(snap.get("device_traces", [[]])[device]):
        a = ev["action"]
        if a == "alloc":
            alive[ev["addr"]] = (ev["size"], site(ev.get("frames")))
            cur += ev["size"]
        elif a in ("free_completed",) and ev["addr"] in alive:
            cur -= alive.pop(ev["addr"])[0]
        elif a == "oom":
            ooms.append((i, ev["size"]))
        series.append(cur)
        if cur > peak:
            peak, peak_i, peak_alive = cur, i, dict(alive)
    at_peak = collections.Counter()
    for size, where in peak_alive.values():
        at_peak[where] += size
    return {"peak": peak, "peak_event": peak_i, "series": series, "at_peak": at_peak.most_common(), "ooms": ooms}


def report(snap: dict, top: int = 5) -> str:
    s, t = summarize(snap, top), timeline(snap)
    out = [f"reserved {fmt(s['reserved'])} | allocated {fmt(s['allocated'])} | reserved but unallocated {fmt(s['unallocated'])} "
           f"({100 * s['fragmentation']:.1f}%) | largest free block {fmt(s['largest_free_block'])} | {s['segments']} segments",
           f"peak allocated during the trace: {fmt(t['peak'])} (event {t['peak_event']})"]
    out.append("alive at the peak, by allocation site:")
    out += [f"  {fmt(n):>12}  {w}" for w, n in t["at_peak"][:top]]
    out.append("live now, by allocation site:")
    out += [f"  {fmt(n):>12}  {w}" for w, n in s["top_sites"]]
    if s["most_wasteful_segments"]:
        out.append("segments with the most unallocated space (pinned by small live blocks):")
        out += [f"  0x{a:x}  {fmt(w)} free of {fmt(tot)}" for a, tot, w in s["most_wasteful_segments"]]
    for i, size in t["ooms"]:
        out.append(f"OOM at event {i}: request of {fmt(size)}")
    return "\n".join(out)
