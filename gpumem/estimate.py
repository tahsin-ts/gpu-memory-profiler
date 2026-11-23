"""Predict GPU memory before launching a job.

Training memory per GPU = model state (weights, gradients, optimizer) / sharding
                        + activations + logits + fixed overhead.
Inference memory        = weights + KV cache + fixed overhead.

Activation formula: Korthikanti et al., "Reducing Activation Recomputation in Large Transformer
Models" (2022): per layer s*b*h*(34 + 5*a*s/h) bytes in 16-bit precision without recomputation;
the 5*a*s/h part is the attention score matrices, which fused (Flash) attention does not store.
It is derived for GPT-style blocks; for SwiGLU MLPs treat it as an approximation.
"""
from dataclasses import dataclass, field

from .model import ModelConfig
from .units import GiB, fmt

BYTES = {"fp32": 4, "bf16": 2, "fp16": 2, "fp8": 1, "int8": 1, "int4": 0.5}


@dataclass
class TrainSetup:
    micro_batch: int = 1
    seq: int = 4096
    precision: str = "bf16"          # mixed precision: 16-bit weights/grads + fp32 master copy
    optimizer: str = "adamw"         # adamw | sgd | adamw8bit
    gpus: int = 1                    # data-parallel ranks the state is sharded over
    zero: int = 0                    # 0 none, 1 optimizer, 2 + gradients, 3 + weights (ZeRO / FSDP)
    recompute: str = "none"          # none | selective | full
    flash_attention: bool = True
    overhead_gib: float = 1.0        # CUDA context, cuBLAS workspace, allocator slack (assumption)


@dataclass
class Breakdown:
    parts: dict = field(default_factory=dict)

    @property
    def total(self) -> float:
        return sum(self.parts.values())

    def table(self) -> str:
        w = max(len(k) for k in self.parts)
        lines = [f"{k:<{w}}  {fmt(v):>12}" for k, v in self.parts.items()]
        return "\n".join(lines + [f"{'TOTAL':<{w}}  {fmt(self.total):>12}"])


def optimizer_bytes_per_param(opt: str, mixed: bool) -> float:
    state = {"adamw": 8, "sgd": 4, "adamw8bit": 2}[opt]        # Adam: m and v in fp32; SGD: momentum
    return state + (4 if mixed else 0)                          # fp32 master weights in mixed precision


def activation_bytes_per_layer(cfg: ModelConfig, s: TrainSetup) -> float:
    sbh = s.seq * s.micro_batch * cfg.hidden
    if s.recompute == "full":
        return 2 * sbh                                          # only the layer input is kept
    scores = 0 if (s.flash_attention or s.recompute == "selective") else 5 * cfg.heads * s.seq / cfg.hidden
    return sbh * (34 + scores)


def train(cfg: ModelConfig, s: TrainSetup) -> Breakdown:
    p = cfg.params()
    mixed = s.precision != "fp32"
    wb = BYTES[s.precision]
    n = max(s.gpus, 1)
    b = Breakdown()
    b.parts["weights"] = p * wb / (n if s.zero >= 3 else 1)
    b.parts["gradients"] = p * wb / (n if s.zero >= 2 else 1)
    b.parts["optimizer"] = p * optimizer_bytes_per_param(s.optimizer, mixed) / (n if s.zero >= 1 else 1)
    b.parts["activations"] = cfg.layers * activation_bytes_per_layer(cfg, s)
    # logits in 16-bit plus an fp32 copy for the loss: often the single largest tensor
    b.parts["logits"] = s.micro_batch * s.seq * cfg.vocab * (wb + 4)
    b.parts["overhead"] = s.overhead_gib * GiB
    return b


def kv_cache_bytes_per_token(cfg: ModelConfig, precision: str = "bf16") -> float:
    return 2 * cfg.layers * cfg.kv_heads * cfg.head_dim * BYTES[precision]     # K and V, every layer


def inference(cfg: ModelConfig, batch: int, context: int, precision: str = "bf16",
              kv_precision: str = "bf16", overhead_gib: float = 1.0) -> Breakdown:
    b = Breakdown()
    b.parts["weights"] = cfg.params() * BYTES[precision]
    b.parts["kv_cache"] = kv_cache_bytes_per_token(cfg, kv_precision) * batch * context
    b.parts["overhead"] = overhead_gib * GiB
    return b


def advise(cfg: ModelConfig, s: TrainSetup, gpu_gib: float) -> list:
    """Changes, in a sensible order, that would make the job fit; each with its predicted total."""
    import copy
    out = []
    cur = copy.copy(s)
    steps = [("flash_attention", True, "use fused (flash) attention"),
             ("recompute", "selective", "selective activation recomputation"),
             ("zero", 1, "shard optimizer state (ZeRO-1)"), ("zero", 2, "also shard gradients (ZeRO-2)"),
             ("zero", 3, "also shard weights (ZeRO-3 / FSDP)"), ("recompute", "full", "full activation recomputation")]
    for attr, val, why in steps:
        if train(cfg, cur).total <= gpu_gib * GiB:
            break
        if attr == "zero" and (cur.gpus <= 1 or cur.zero >= val):
            continue
        if getattr(cur, attr) == val:
            continue
        before = train(cfg, cur).total
        setattr(cur, attr, val)
        if train(cfg, cur).total < before:            # only report changes that actually save memory
            out.append((why, train(cfg, cur).total))
    while train(cfg, cur).total > gpu_gib * GiB and cur.micro_batch > 1:
        cur.micro_batch //= 2
        out.append((f"halve micro-batch to {cur.micro_batch} (raise gradient accumulation to keep the global batch)", train(cfg, cur).total))
    return out
