from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# Gate decision constants — shared with anomaly_detector.py
ACCEPT = "accept"
FLAG = "flag"
QUARANTINE = "quarantine"


class TrustState:
    """
    Per-client trust scores and quarantine state.

    Trust score T_i ∈ (0, 1], initialized to 1.0:
        accept     → T_i = min(1.0, T_i + 0.05)
        flag       → T_i = T_i × 0.9
        quarantine → T_i = T_i × 0.5, client excluded from aggregation

    A quarantined client re-enters after K_rehab consecutive clean (accept) rounds.
    """

    def __init__(
        self,
        client_ids: list[int],
        rehab_rounds: int = 3,
    ) -> None:
        self.scores: dict[int, float] = {cid: 1.0 for cid in client_ids}
        self.quarantine_set: set[int] = set()
        self.rehab_counts: dict[int, int] = {cid: 0 for cid in client_ids}
        self.K_rehab = rehab_rounds

    def update(self, client_id: int, gate_decision: str) -> None:
        """Apply gate decision to trust score and quarantine state."""
        score = self.scores.get(client_id, 1.0)

        if gate_decision == ACCEPT:
            self.scores[client_id] = min(1.0, score + 0.05)
            if client_id in self.quarantine_set:
                self.rehab_counts[client_id] = self.rehab_counts.get(client_id, 0) + 1
                if self.rehab_counts[client_id] >= self.K_rehab:
                    self.quarantine_set.discard(client_id)
                    self.rehab_counts[client_id] = 0
                    logger.info("Client %d rehabilitated after %d clean rounds.", client_id, self.K_rehab)
            else:
                self.rehab_counts[client_id] = 0

        elif gate_decision == FLAG:
            self.scores[client_id] = score * 0.9
            self.rehab_counts[client_id] = 0

        elif gate_decision == QUARANTINE:
            self.scores[client_id] = score * 0.5
            self.rehab_counts[client_id] = 0
            if client_id not in self.quarantine_set:
                logger.warning("Client %d quarantined (trust score now %.3f).", client_id, self.scores[client_id])
            self.quarantine_set.add(client_id)

        else:
            raise ValueError(f"Unknown gate decision: {gate_decision!r}")

    def is_quarantined(self, client_id: int) -> bool:
        return client_id in self.quarantine_set

    def get_score(self, client_id: int) -> float:
        return self.scores.get(client_id, 1.0)

    def snapshot(self) -> dict:
        return {
            "trust_scores": dict(self.scores),
            "quarantined": list(self.quarantine_set),
            "rehab_counts": dict(self.rehab_counts),
        }
