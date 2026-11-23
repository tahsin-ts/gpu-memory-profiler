"""gpumem command line."""
import argparse
import sys
import time

from . import estimate as E
from .allocator import CachingAllocator
from .model import PRESETS
from .snapshot import load, report
from .units import GiB, MiB, fmt
from .workloads import varlen_training


def main(argv=None):
    ap = argparse.ArgumentParser(prog="gpumem", description="Predict, simulate and explain GPU memory.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("estimate", help="memory needed by a training or inference job")
    e.add_argument("--model", choices=sorted(PRESETS), required=True)
    e.add_argument("--mode", choices=["train", "infer"], default="train")
    e.add_argument("--micro-batch", type=int, default=1)
    e.add_argument("--seq", type=int, default=4096)
    e.add_argument("--precision", default="bf16", choices=sorted(E.BYTES))
    e.add_argument("--optimizer", default="adamw", choices=["adamw", "sgd", "adamw8bit"])
    e.add_argument("--gpus", type=int, default=1)
    e.add_argument("--zero", type=int, default=0, choices=[0, 1, 2, 3])
    e.add_argument("--recompute", default="none", choices=["none", "selective", "full"])
    e.add_argument("--no-flash", action="store_true")
    e.add_argument("--batch", type=int, default=1, help="inference: concurrent sequences")
    e.add_argument("--kv-precision", default="bf16", choices=sorted(E.BYTES))
    e.add_argument("--gpu-gib", type=float, default=80.0)

    s = sub.add_parser("simulate", help="run a workload through the caching-allocator simulator")
    s.add_argument("--capacity-gib", type=float, default=6.0)
    s.add_argument("--max-split-mb", type=int, default=0)
    s.add_argument("--steps", type=int, default=300)
    s.add_argument("--snapshot", help="write the final snapshot (pickle) here")

    a = sub.add_parser("analyze", help="explain a snapshot from torch or from `simulate --snapshot`")
    a.add_argument("path")
    a.add_argument("--top", type=int, default=5)

    w = sub.add_parser("watch", help="GPU memory per device and process via NVML")
    w.add_argument("--interval", type=float, default=1.0)
    w.add_argument("--count", type=int, default=0, help="0 = forever")

    args = ap.parse_args(argv)
    if args.cmd == "estimate":
        cfg = PRESETS[args.model]
        print(f"{cfg.name}: {cfg.params():,} parameters")
        if args.mode == "infer":
            b = E.inference(cfg, args.batch, args.seq, args.precision, args.kv_precision)
            print(f"KV cache per token: {fmt(E.kv_cache_bytes_per_token(cfg, args.kv_precision))}")
            print(b.table())
        else:
            st = E.TrainSetup(args.micro_batch, args.seq, args.precision, args.optimizer, args.gpus, args.zero,
                              args.recompute, not args.no_flash)
            b = E.train(cfg, st)
            print(b.table())
            fits = b.total <= args.gpu_gib * GiB
            print(f"{'fits' if fits else 'does NOT fit'} on a {args.gpu_gib:g} GiB GPU")
            if not fits:
                for why, total in E.advise(cfg, st, args.gpu_gib):
                    print(f"  -> {why}: {fmt(total)}")
        return 0
    if args.cmd == "simulate":
        al = CachingAllocator(int(args.capacity_gib * GiB), args.max_split_mb * MiB or None)
        steps, err = varlen_training(al, steps=args.steps)
        st = al.stats()
        print(f"completed {steps}/{args.steps} steps | peak allocated {fmt(st['peak_allocated'])} | "
              f"peak reserved {fmt(st['peak_reserved'])} | allocation retries {st['num_alloc_retries']} | OOMs {st['num_ooms']}")
        if err:
            print(err)
        if args.snapshot:
            import pickle
            with open(args.snapshot, "wb") as f:
                pickle.dump(al.snapshot(), f)
        return 1 if err else 0
    if args.cmd == "analyze":
        print(report(load(args.path), args.top))
        return 0
    if args.cmd == "watch":
        from .nvml import NVML
        try:
            nv = NVML()
        except OSError:
            print("NVML not found: this machine has no NVIDIA driver", file=sys.stderr)
            return 2
        n = 0
        while True:
            for i in range(nv.count()):
                m = nv.memory(i)
                procs = ", ".join(f"pid {p} {fmt(u)}" for p, u in nv.processes(i))
                print(f"GPU {i}: {fmt(m['used'])} / {fmt(m['total'])} used   {procs}")
            n += 1
            if args.count and n >= args.count:
                return 0
            time.sleep(args.interval)


if __name__ == "__main__":
    sys.exit(main())
