import time, torch
from pmsa.models.detector import LRTSurrogate

def measure_latency_ms(fn, *args, warmup=10, iters=100, **kwargs):
    for _ in range(warmup):
        fn(*args, **kwargs)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(iters):
        fn(*args, **kwargs)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return (time.time() - t0) * 1000.0 / iters

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = LRTSurrogate().to(device).eval()
    x = torch.randn(1, 3, 64, 64, device=device)
    with torch.no_grad():
        ms = measure_latency_ms(model, x)
    print(f"Detector forward latency: {ms:.2f} ms on {device}")
