import torch
import time

ITERS = 10000
SHAPE = (1000, 1000)


def where_arithmetic(cond, x, y):
    """
    Custom implementation of torch.where using arithmetic operations.
    """
    # Convert condition to float
    cond_float = cond.float()
    # Use arithmetic to select values based on condition
    return cond_float * x + (1 - cond_float) * y


def main():
    devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else []) + (["mps"] if torch.backends.mps.is_available() else [])
    tens = {device: torch.randn(SHAPE, device=device) for device in devices}
    masks = {device: torch.randint(0, 2, SHAPE, device=device).bool() for device in devices}
    for device in devices:
        data = tens[device]
        mask = masks[device]
        start = time.time()
        for _ in range(ITERS):
            selected = torch.where(mask, data, data)
        end = time.time()
        print(f"torch.where on {device}: {end - start:.4f} seconds")
        start = time.time()
        for _ in range(ITERS):
            selected_arithmetic = where_arithmetic(mask, data, data)
        end = time.time()
        print(f"Custom where on {device}: {end - start:.4f} seconds")
        if not torch.allclose(selected, selected_arithmetic):
            print(f"Mismatch between torch.where and custom where on {device}")
        print(f"Results match on {device}")
        print("-" * 40)


if __name__ == "__main__":
    main()
