"""Record a real PyTorch job's allocations and analyse them (needs torch + a CUDA GPU).

    python examples/record_real_job.py && gpumem analyze /tmp/train.pickle
"""
import torch

from gpumem.torch_hooks import record, step_stats

model = torch.nn.Sequential(*[torch.nn.Linear(4096, 4096) for _ in range(8)]).cuda()
opt = torch.optim.AdamW(model.parameters())
with record("/tmp/train.pickle"):
    for step in range(5):
        x = torch.randn(64, 4096, device="cuda")
        loss = model(x).pow(2).mean()
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        print(step, step_stats())
