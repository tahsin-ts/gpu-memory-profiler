"""Transformer shapes and exact parameter counts."""
from dataclasses import dataclass


@dataclass(frozen=True)
class ModelConfig:
    name: str
    hidden: int
    layers: int
    heads: int
    kv_heads: int
    intermediate: int
    vocab: int
    gated_mlp: bool = True          # SwiGLU (Llama): 3 matrices; GPT-2: 2 matrices
    tied_embeddings: bool = False
    bias: bool = False              # GPT-2 has biases and LayerNorm biases
    max_positions: int = 0          # learned position embeddings (GPT-2); 0 = rotary (none)

    @property
    def head_dim(self) -> int:
        return self.hidden // self.heads

    def params(self) -> int:
        h, kv = self.hidden, self.kv_heads * self.head_dim
        attn = h * h + 2 * h * kv + h * h                     # q, k, v, o
        mlp = (3 if self.gated_mlp else 2) * h * self.intermediate
        norms = (4 if self.bias else 2) * h                   # 2 norms per layer (+ biases)
        if self.bias:
            attn += h + 2 * kv + h
            mlp += self.intermediate + h
        per_layer = attn + mlp + norms
        embed = self.vocab * h + self.max_positions * h
        head = 0 if self.tied_embeddings else self.vocab * h
        final_norm = (2 if self.bias else 1) * h
        return embed + self.layers * per_layer + final_norm + head


PRESETS = {
    "gpt2": ModelConfig("gpt2", 768, 12, 12, 12, 3072, 50257, gated_mlp=False, tied_embeddings=True, bias=True, max_positions=1024),
    "llama2-7b": ModelConfig("llama2-7b", 4096, 32, 32, 32, 11008, 32000),
    "llama2-13b": ModelConfig("llama2-13b", 5120, 40, 40, 40, 13824, 32000),
    "llama3-8b": ModelConfig("llama3-8b", 4096, 32, 32, 8, 14336, 128256),
    "llama3-70b": ModelConfig("llama3-70b", 8192, 80, 64, 8, 28672, 128256),
}
