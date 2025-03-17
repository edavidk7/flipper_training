import torch
import time

ITERS = 10000
SHAPE = (1000, 1000)


def clamp_arithmetic(x, min_val, max_val):
    min_mask = x < min_val
    max_mask = x > max_val
    combined_mask = min_mask | max_mask
    return min_mask.float() * min_val + max_mask.float() * max_val + (1.0 - combined_mask.float()) * x


def main():
    devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else []) + (["mps"] if torch.backends.mps.is_available() else [])
    tens = {device: torch.randn(SHAPE, device=device) for device in devices}
    min_val = torch.randn(1, device=devices[0])
    max_val = torch.randn(1, device=devices[0])
    for device in devices:
        data = tens[device]
        start = time.time()
        for _ in range(ITERS):
            clamped = torch.clamp(data, min_val, max_val)
        end = time.time()
        print(f"torch.clamp on {device}: {end - start:.4f} seconds")
        start = time.time()
        for _ in range(ITERS):
            clamped_arithmetic = clamp_arithmetic(data, min_val, max_val)
        end = time.time()
        print(f"Custom clamp on {device}: {end - start:.4f} seconds")
        if not torch.allclose(clamped, clamped_arithmetic):
            print(f"Mismatch between torch.clamp and custom clamp on {device}")
        print(f"Results match on {device}")
        print("-" * 40)


if __name__ == "__main__":
    main()
