import pytest

from gpumem.nvml import NVML


def test_nvml_on_a_gpu_machine():
    try:
        nv = NVML()
    except OSError:
        pytest.skip("no NVIDIA driver on this machine")
    assert nv.count() >= 1
    m = nv.memory(0)
    assert m["total"] >= m["used"] > 0
    nv.close()
