import hashlib
import random

def get_gstin_seed(gstin: str) -> int:
    """Generate a deterministic seed from a GSTIN string."""
    return int(hashlib.md5(gstin.encode('utf-8')).hexdigest(), 16) % (10**8)

def deterministic_random(seed: int):
    """Return a seeded Random instance."""
    return random.Random(seed)

def normalize(value: float, min_val: float, max_val: float) -> float:
    """Min-max noramlization between 0 and 1."""
    if max_val == min_val:
        return 0.0
    return max(0.0, min(1.0, (value - min_val) / (max_val - min_val)))
