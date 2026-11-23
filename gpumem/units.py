KiB, MiB, GiB = 2**10, 2**20, 2**30


def fmt(n: float) -> str:
    """Human-readable bytes, the way PyTorch prints them (binary units)."""
    for unit, size in (("GiB", GiB), ("MiB", MiB), ("KiB", KiB)):
        if abs(n) >= size:
            return f"{n / size:.2f} {unit}"
    return f"{int(n)} bytes"
