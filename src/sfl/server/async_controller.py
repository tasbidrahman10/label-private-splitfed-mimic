from __future__ import annotations

import asyncio
import logging

logger = logging.getLogger(__name__)


class AsyncRoundController:
    """
    Replaces the synchronous 'wait for all N clients' FedAvg trigger.

    Opens a timed window per round. When the deadline arrives, aggregation
    proceeds with whatever clients have submitted — even if fewer than N.

    The window duration adapts slightly upward when average staleness is high,
    giving perpetually-lagging clients a small grace extension.

    Usage (in app.py):
        controller.receive_submission(client_id, weights, sample_count)
        if not controller.round_open:
            asyncio.create_task(run_async_round(server_state))
    """

    def __init__(
        self,
        client_ids: list[int],
        base_window_seconds: float = 60.0,
        max_consecutive_misses: int = 3,
    ) -> None:
        self.client_ids: set[int] = set(client_ids)
        self.T_base = base_window_seconds
        self.K_max = max_consecutive_misses

        self.submission_buffer: dict[int, dict] = {}
        self.sample_count_buffer: dict[int, int] = {}
        self.consecutive_misses: dict[int, int] = {cid: 0 for cid in client_ids}
        self.round_number: int = 0
        self.round_open: bool = False

    def receive_submission(
        self, client_id: int, weights: dict, sample_count: int
    ) -> bool:
        """
        Register a client encoder submission.

        Returns True if the submission was accepted (window open),
        False if the window is closed (client is late for this round).
        """
        if not self.round_open:
            logger.warning(
                "Client %d submitted outside the window (round %d closed).",
                client_id, self.round_number,
            )
            return False
        if client_id not in self.client_ids:
            logger.warning("Unknown client %d — ignoring submission.", client_id)
            return False
        self.submission_buffer[client_id] = weights
        self.sample_count_buffer[client_id] = sample_count
        logger.debug(
            "Round %d: received submission from client %d (%d/%d clients).",
            self.round_number, client_id,
            len(self.submission_buffer), len(self.client_ids),
        )
        return True

    def open_round_immediately(self) -> None:
        """
        Mark the round as open and reset submission buffers.

        Called synchronously (inside the lock) when the first submission of a
        round arrives, before spawning the background deadline task. This
        ensures the window is open before `receive_submission` is called for
        the triggering client.
        """
        self.round_open = True
        self.submission_buffer = {}
        self.sample_count_buffer = {}
        logger.info("Round %d window opened.", self.round_number)

    async def wait_for_deadline(self, staleness_counts: dict[int, int]) -> dict[int, dict]:
        """
        Sleep for T_adaptive seconds then close the window and return all
        submissions received. Called as an asyncio background task.

        T_adaptive = T_base × (1 + 0.1 × mean_staleness)
        so rounds with many stale clients get a slightly longer window.
        """
        mean_staleness = (
            sum(staleness_counts.values()) / len(staleness_counts)
            if staleness_counts else 0.0
        )
        T_adaptive = self.T_base * (1.0 + 0.1 * mean_staleness)
        logger.info(
            "Round %d sleeping %.1fs (mean staleness=%.2f).",
            self.round_number, T_adaptive, mean_staleness,
        )

        await asyncio.sleep(T_adaptive)

        self.round_open = False
        self._update_miss_counts()
        self.round_number += 1

        logger.info(
            "Round %d closed: %d/%d clients submitted. Miss counts: %s",
            self.round_number - 1,
            len(self.submission_buffer),
            len(self.client_ids),
            self.consecutive_misses,
        )
        return dict(self.submission_buffer)

    def get_sample_counts(self) -> dict[int, int]:
        """Return sample counts for clients that submitted this round."""
        return dict(self.sample_count_buffer)

    def _update_miss_counts(self) -> None:
        for cid in self.client_ids:
            if cid in self.submission_buffer:
                self.consecutive_misses[cid] = 0
            else:
                self.consecutive_misses[cid] += 1
                if self.consecutive_misses[cid] >= self.K_max:
                    logger.warning(
                        "Client %d has missed %d consecutive rounds — flagging for inspection.",
                        cid, self.consecutive_misses[cid],
                    )
