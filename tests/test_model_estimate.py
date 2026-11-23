import pytest

from gpumem import estimate as E
from gpumem.model import PRESETS
from gpumem.units import GiB, KiB, MiB


@pytest.mark.parametrize("name,params", [
    ("gpt2", 124_439_808),            # Hugging Face gpt2
    ("llama2-7b", 6_738_415_616),     # Llama-2-7b-hf
    ("llama2-13b", 13_015_864_320),
    ("llama3-8b", 8_030_261_248),     # Meta-Llama-3-8B
    ("llama3-70b", 70_553_706_496),
])
def test_parameter_counts_are_exact(name, params):
    assert PRESETS[name].params() == params


def test_kv_cache_per_token():
    assert E.kv_cache_bytes_per_token(PRESETS["llama2-7b"]) == 512 * KiB     # 32 layers x 32 heads x 128 x 2 x 2 B
    assert E.kv_cache_bytes_per_token(PRESETS["llama3-8b"]) == 128 * KiB     # grouped-query attention: 8 KV heads
    assert E.kv_cache_bytes_per_token(PRESETS["llama3-8b"], "fp8") == 64 * KiB


def test_mixed_precision_adam_is_16_bytes_per_param():
    cfg = PRESETS["llama2-7b"]
    b = E.train(cfg, E.TrainSetup())
    state = b.parts["weights"] + b.parts["gradients"] + b.parts["optimizer"]
    assert state == 16 * cfg.params()


def test_zero_stages_shard_the_right_parts():
    cfg = PRESETS["llama2-7b"]
    base = E.train(cfg, E.TrainSetup(gpus=8)).parts
    z1 = E.train(cfg, E.TrainSetup(gpus=8, zero=1)).parts
    z3 = E.train(cfg, E.TrainSetup(gpus=8, zero=3)).parts
    assert z1["optimizer"] == base["optimizer"] / 8 and z1["weights"] == base["weights"]
    assert z3["weights"] == base["weights"] / 8 and z3["gradients"] == base["gradients"] / 8
    assert z3["activations"] == base["activations"]            # sharding never shrinks activations


def test_activation_formula_and_flash():
    cfg = PRESETS["gpt2"]
    s = E.TrainSetup(micro_batch=1, seq=1024, flash_attention=False)
    sbh = 1024 * 768
    assert E.activation_bytes_per_layer(cfg, s) == pytest.approx(sbh * (34 + 5 * 12 * 1024 / 768))
    s.flash_attention = True
    assert E.activation_bytes_per_layer(cfg, s) == sbh * 34
    s.recompute = "full"
    assert E.activation_bytes_per_layer(cfg, s) == 2 * sbh


def test_advice_reaches_a_fitting_plan():
    cfg = PRESETS["llama2-7b"]
    steps = E.advise(cfg, E.TrainSetup(seq=4096, gpus=8), 80)
    assert steps and steps[-1][1] <= 80 * GiB
    assert all(b < a for (_, a), (_, b) in zip(steps, steps[1:]))      # every step saves memory


def test_inference_breakdown():
    b = E.inference(PRESETS["llama3-8b"], batch=8, context=8192)
    assert b.parts["kv_cache"] == 8 * 8192 * 128 * KiB == 8 * GiB
    assert b.parts["weights"] == PRESETS["llama3-8b"].params() * 2
