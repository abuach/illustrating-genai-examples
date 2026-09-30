"""Privacy-preserving computation primitives: differential privacy and
federated learning.

These are the math primitives the Responsible AI chapter builds on. The
chapter's security demos (attacks, redaction, access control, moderation,
watermarking, governance) live in :mod:`genai.security`.
"""
import numpy as np


# ── Differential Privacy ─────────────────────────────────────────────────────

def add_noise(value: float, sensitivity: float = 1.0, epsilon: float = 1.0) -> float:
    """Add Laplace noise to a value for differential privacy."""
    return value + np.random.laplace(0, sensitivity / epsilon)


def private_avg(data, lo: float = 0, hi: float = 100, epsilon: float = 1.0) -> float:
    """Return a differentially private average of data in [lo, hi]."""
    clipped = np.clip(data, lo, hi)
    sensitivity = (hi - lo) / len(clipped)
    return float(np.mean(clipped)) + float(np.random.laplace(0, sensitivity / epsilon))


# ── Federated Learning ────────────────────────────────────────────────────────

def federated_avg(updates: list) -> np.ndarray:
    """Average a list of numpy weight vectors (the FedAvg algorithm)."""
    return np.mean(updates, axis=0)


def hospital_bp_cohorts(rng):
    """Return three simulated hospital cohorts for the federated-learning demo."""
    return {
        "Valley Medical":  rng.normal(130, 15, 120),
        "River Clinic":    rng.normal(136, 18,  85),
        "Mountain Health": rng.normal(128, 12, 200),
    }
