from __future__ import annotations

import asyncio
import logging
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel

from sfl.common.config import load_yaml
from sfl.common.serialization import base64_to_state_dict, state_dict_to_base64
from sfl.server.state import ClientRegistration, ServerState
from sfl.server.trainer import (
    option_b_backward,
    option_b_forward,
    predict_from_activation,
)
from sfl.server.trust_state import QUARANTINE

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pydantic request/response models (unchanged)
# ---------------------------------------------------------------------------

class RegisterRequest(BaseModel):
    client_id: int
    client_name: str
    train_size: int
    pos_weight: float


class PredictRequest(BaseModel):
    activation: list[Any]


class OptionBForwardRequest(BaseModel):
    client_id: int
    activation: list[Any]
    round: int | None = None
    batch: int | None = None


class OptionBBackwardRequest(BaseModel):
    client_id: int
    context_id: str
    logit_gradient: list[Any]


class EncoderSubmitRequest(BaseModel):
    client_id: int
    sample_count: int
    state_dict: str


class SaveServerRequest(BaseModel):
    filename: str = "server_model_latest.pth"


# ---------------------------------------------------------------------------
# Async round background task
# ---------------------------------------------------------------------------

async def _run_async_round(state: ServerState) -> None:
    """
    Background task: waits for the submission window to close, then runs
    SNAS anomaly detection and RobustAsyncFedAvg aggregation.

    Called via asyncio.create_task() when the first encoder submission of a
    round arrives. The window duration is controlled by AsyncRoundController.
    """
    try:
        submitted = await state.async_controller.wait_for_deadline(
            state.staleness_registry.counts
        )
    except Exception:
        logger.exception("Exception in wait_for_deadline — restoring previous encoder.")
        async with state.lock:
            state.global_encoder = state.prev_global_encoder
        return

    try:
        await _process_round(state, submitted)
    except Exception:
        logger.exception(
            "Exception in round %d aggregation — restoring previous encoder so clients don't timeout.",
            state.current_round,
        )
        async with state.lock:
            if state.global_encoder is None:
                state.global_encoder = state.prev_global_encoder
            state.current_round += 1


async def _process_round(state: ServerState, submitted: dict) -> None:
    """Core aggregation logic, separated so exceptions can be caught cleanly."""
    async with state.lock:
        round_idx = state.current_round

        if not submitted:
            logger.warning("Round %d: no submissions received — skipping aggregation.", round_idx)
            state.staleness_registry.update_after_round(set())
            state.current_round += 1
            return

        sample_counts = state.async_controller.get_sample_counts()
        gate_decisions: dict[int, str] = {}
        weight_log: dict[int, dict] = {}

        # Compute SNAS and gate decisions for each submitted client
        for cid, weights in submitted.items():
            activations = state.activation_buffer.get(cid)
            tau = state.staleness_registry.get_staleness(cid)

            if activations is not None:
                snas_info = state.gate.snas_metrics(
                    cid, activations, weights, state.prev_global_encoder, tau
                )
                snas = snas_info["snas"] or 0.0
            else:
                snas_info = {
                    "activation_divergence": None,
                    "cosine_anomaly": None,
                    "norm_anomaly": None,
                    "snas": None,
                    "staleness_tau": tau,
                }
                snas = 0.0

            decision = state.gate.gate(cid, snas, state.current_round)
            gate_decisions[cid] = decision
            state.trust.update(cid, decision)

            # Update activation reference only for clients that passed the gate
            if decision != QUARANTINE and activations is not None:
                state.gate.update_reference(cid, activations)

            weight_log[cid] = {
                **snas_info,
                "gate_decision": decision,
                "trust_score": state.trust.get_score(cid),
                "included_in_aggregation": decision != QUARANTINE,
            }

        # Robust async FedAvg (with norm clipping against prev round encoder)
        new_global = state.robust_fedavg.aggregate(
            submitted,
            state.staleness_registry.counts,
            gate_decisions,
            sample_counts,
            device=state.device,
            prev_global_encoder=state.prev_global_encoder,
        )

        # Per-client aggregation weight logging
        agg_weights = state.robust_fedavg.log_weights(
            submitted, state.staleness_registry.counts, gate_decisions, sample_counts
        )

        if new_global is not None:
            state.global_encoder = new_global
            logger.info("Round %d: global encoder updated from %d client(s).", round_idx, len(submitted))
        else:
            # All submitted clients quarantined — fall back to previous round's encoder
            # so clients don't timeout waiting for a 404 that will never resolve.
            state.global_encoder = state.prev_global_encoder
            logger.warning(
                "Round %d: all clients quarantined — restored previous round's global encoder.", round_idx
            )

        # Log per-client round metrics
        for cid in submitted:
            row = {
                "round": round_idx,
                "client_id": cid,
                **weight_log[cid],
                **agg_weights.get(cid, {"aggregation_weight_normalized": 0.0}),
            }
            state.append_metric(row)

        state.staleness_registry.update_after_round(set(submitted.keys()))
        state.current_round += 1


# ---------------------------------------------------------------------------
# FastAPI application factory
# ---------------------------------------------------------------------------

def create_app(config_path: str = "configs/server.yaml") -> FastAPI:
    config = load_yaml(config_path)
    server_state = ServerState(config)
    app = FastAPI(title="MIMIC SplitFed Server (Async-SplitFed-IR)", version="0.2.0")

    async def require_token(authorization: str | None = Header(default=None)) -> None:
        expected = server_state.expected_token()
        if not expected:
            return
        if authorization != f"Bearer {expected}":
            raise HTTPException(status_code=401, detail="Invalid or missing bearer token")

    @app.get("/health")
    async def health() -> dict:
        return {
            "status": "ok",
            "device": str(server_state.device),
            "registered_clients": len(server_state.clients),
            "current_round": server_state.async_controller.round_number,
            "window_open": server_state.async_controller.round_open,
        }

    @app.post("/register_client", dependencies=[Depends(require_token)])
    async def register_client(request: RegisterRequest) -> dict:
        async with server_state.lock:
            server_state.register_client(
                ClientRegistration(
                    client_id=request.client_id,
                    client_name=request.client_name,
                    train_size=request.train_size,
                    pos_weight=request.pos_weight,
                )
            )
        return {"status": "registered", "client_id": request.client_id}

    @app.post("/predict", dependencies=[Depends(require_token)])
    async def predict(request: PredictRequest) -> dict:
        async with server_state.lock:
            return predict_from_activation(server_state, request.activation)

    @app.post("/option-b/forward", dependencies=[Depends(require_token)])
    async def option_b_forward_endpoint(request: OptionBForwardRequest) -> dict:
        async with server_state.lock:
            return option_b_forward(
                server_state,
                client_id=request.client_id,
                activation_payload=request.activation,
                round_idx=request.round,
                batch_idx=request.batch,
            )

    @app.post("/option-b/backward", dependencies=[Depends(require_token)])
    async def option_b_backward_endpoint(request: OptionBBackwardRequest) -> dict:
        async with server_state.lock:
            try:
                return option_b_backward(
                    server_state,
                    client_id=request.client_id,
                    context_id=request.context_id,
                    logit_gradient_payload=request.logit_gradient,
                )
            except KeyError as exc:
                raise HTTPException(status_code=404, detail="Unknown Option B context") from exc
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/fedavg/open_round", dependencies=[Depends(require_token)])
    async def open_fedavg_round() -> dict:
        """
        Explicitly open the async submission window for the current round.
        Called by the runner before clients start submitting, so that late
        submissions find a closed window and are correctly rejected rather
        than accidentally opening a new round.
        """
        async with server_state.lock:
            if server_state.async_controller.round_open:
                return {"status": "already_open", "round": server_state.async_controller.round_number}
            server_state.prev_global_encoder = server_state.global_encoder  # save for SNAS
            server_state.global_encoder = None  # clear so clients get 404 during window
            server_state.async_controller.open_round_immediately()
            asyncio.create_task(_run_async_round(server_state))
        return {"status": "opened", "round": server_state.async_controller.round_number}

    @app.post("/fedavg/submit_encoder", dependencies=[Depends(require_token)])
    async def submit_encoder(request: EncoderSubmitRequest) -> dict:
        async with server_state.lock:
            state_dict = base64_to_state_dict(request.state_dict, server_state.device)
            accepted = server_state.async_controller.receive_submission(
                request.client_id, state_dict, request.sample_count
            )
        return {
            "status": "submitted" if accepted else "rejected_late",
            "accepted": accepted,
            "round": server_state.async_controller.round_number,
            "window_open": server_state.async_controller.round_open,
        }

    @app.get("/fedavg/global_encoder", dependencies=[Depends(require_token)])
    async def get_global_encoder() -> dict:
        async with server_state.lock:
            if server_state.global_encoder is None:
                raise HTTPException(status_code=404, detail="No global encoder available yet")
            payload = state_dict_to_base64(server_state.global_encoder)
        return {"state_dict": payload}

    @app.get("/metrics", dependencies=[Depends(require_token)])
    async def metrics() -> dict:
        return {"metrics": server_state.metrics[-200:]}

    @app.post("/checkpoint/server", dependencies=[Depends(require_token)])
    async def save_server_checkpoint(request: SaveServerRequest) -> dict:
        async with server_state.lock:
            path = server_state.save_checkpoint(request.filename)
        return {"status": "saved", "path": str(path)}

    return app


app = create_app()
