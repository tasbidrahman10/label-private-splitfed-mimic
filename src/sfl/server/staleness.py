from __future__ import annotations

import math


class StalenessDecay:
    """
    Four staleness decay functions mapping τ (missed rounds) → weight in [0, 1].

    All return 1.0 when τ=0 (client submitted on time) and decrease toward 0
    as τ increases. Used in RobustAsyncFedAvg to downweight stale updates.
    """

    @staticmethod
    def linear(tau: int, alpha: float = 1.0) -> float:
        """w = 1 / (1 + alpha * tau)"""
        return 1.0 / (1.0 + alpha * tau)

    @staticmethod
    def exponential(tau: int, alpha: float = 0.5) -> float:
        """w = exp(-alpha * tau)"""
        return math.exp(-alpha * tau)

    @staticmethod
    def polynomial(tau: int, beta: float = 2.0) -> float:
        """w = 1 / (1 + tau^beta)"""
        return 1.0 / (1.0 + tau ** beta)

    @staticmethod
    def step(tau: int, cutoff: int = 3) -> float:
        """w = 1 if tau <= cutoff else 0"""
        return 1.0 if tau <= cutoff else 0.0

    # Map name → callable for config-driven selection
    REGISTRY: dict[str, staticmethod] = {}

    @classmethod
    def get(cls, name: str):
        """Return decay function by name: 'linear', 'exponential', 'polynomial', 'step'."""
        mapping = {
            "linear": cls.linear,
            "exponential": cls.exponential,
            "polynomial": cls.polynomial,
            "step": cls.step,
        }
        if name not in mapping:
            raise ValueError(f"Unknown decay function '{name}'. Choose from {list(mapping)}")
        return mapping[name]


class StalenessRegistry:
    """
    Tracks per-client staleness counter τ_i.

    τ_i(r) = number of consecutive rounds before round r in which client i
    did NOT submit an encoder update.

    On time → τ resets to 0.
    Missed round → τ increments by 1.
    """

    def __init__(self, client_ids: list[int]) -> None:
        self.counts: dict[int, int] = {cid: 0 for cid in client_ids}

    def get_staleness(self, client_id: int) -> int:
        return self.counts.get(client_id, 0)

    def update_after_round(self, submitted_ids: set[int]) -> None:
        """Call after each round closes. Increments τ for missing clients, resets for submitted."""
        for cid in self.counts:
            if cid in submitted_ids:
                self.counts[cid] = 0
            else:
                self.counts[cid] += 1

    def all_counts(self) -> dict[int, int]:
        return dict(self.counts)
