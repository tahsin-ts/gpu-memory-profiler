"""Read GPU memory straight from the NVIDIA driver (NVML) with ctypes: no extra packages.
Shows what nvidia-smi shows, including memory used by OTHER processes, which PyTorch cannot see."""
import ctypes


class _Memory(ctypes.Structure):
    _fields_ = [("total", ctypes.c_ulonglong), ("free", ctypes.c_ulonglong), ("used", ctypes.c_ulonglong)]


class _Process(ctypes.Structure):          # nvmlProcessInfo_v1_t: the layout that every driver accepts
    _fields_ = [("pid", ctypes.c_uint), ("usedGpuMemory", ctypes.c_ulonglong)]


class NVML:
    def __init__(self, lib="libnvidia-ml.so.1"):
        self.lib = ctypes.CDLL(lib)        # raises OSError when there is no NVIDIA driver
        self._check(self.lib.nvmlInit_v2())

    def _check(self, rc):
        if rc != 0:
            self.lib.nvmlErrorString.restype = ctypes.c_char_p
            raise RuntimeError(f"NVML error {rc}: {self.lib.nvmlErrorString(rc).decode()}")

    def count(self) -> int:
        n = ctypes.c_uint()
        self._check(self.lib.nvmlDeviceGetCount_v2(ctypes.byref(n)))
        return n.value

    def _handle(self, i):
        h = ctypes.c_void_p()
        self._check(self.lib.nvmlDeviceGetHandleByIndex_v2(i, ctypes.byref(h)))
        return h

    def memory(self, i: int) -> dict:
        m = _Memory()
        self._check(self.lib.nvmlDeviceGetMemoryInfo(self._handle(i), ctypes.byref(m)))
        return {"total": m.total, "used": m.used, "free": m.free}

    def processes(self, i: int) -> list:
        n = ctypes.c_uint(64)
        arr = (_Process * 64)()
        self._check(self.lib.nvmlDeviceGetComputeRunningProcesses(self._handle(i), ctypes.byref(n), arr))
        return [(arr[k].pid, arr[k].usedGpuMemory) for k in range(n.value)]

    def close(self):
        self.lib.nvmlShutdown()
