from __future__ import annotations

import asyncio
import logging
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel

from sfl.common.config import load_yaml
from sfl.common.serialization import base64_to_state_dict, state_dict_to_base64
from sfl.server.baseline_aggregators import (
    flat_to_state_dict, fltrust_aggregate, krum_aggregate,
    state_dict_to_flat, trimmed_mean_aggregate,
)
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


class ResetRequest(BaseModel):
    config_overrides: dict = {}  # e.g. {"aggregator_type": "krum", "snas_gamma": 0.5}


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


async def _process_round_baseline(
    state: ServerState,
    submitted: dict,
    aggregator: str,
) -> None:
    """
    Aggregation path for baseline methods (Krum, Trimmed Mean, FLTrust).
    These methods have NO staleness awareness and NO anomaly gating — this
    is intentional and is the key limitation documented in the paper comparison.
    """
    async with state.lock:
        round_idx = state.current_round

        if not submitted:
            logger.warning("Baseline %s round %d: no submissions.", aggregator, round_idx)
            state.global_encoder = state.prev_global_encoder
            state.current_round += 1
            return

        sample_counts = state.async_controller.get_sample_counts()

        # Flatten all submitted state_dicts to numpy vectors for baseline aggregators
        flat_submissions: dict[int, "np.ndarray"] = {
            cid: state_dict_to_flat(w) for cid, w in submitted.items()
        }

        # Reference state_dict structure (for unflattening the result)
        ref_sd = next(iter(submitted.values()))

        new_flat: "np.ndarray | None" = None
        selected_client: int | None = None

        if aggregator == "krum":
            new_flat, selected_client = krum_aggregate(flat_submissions, f_byzantine=1)
            logger.info("Krum selected client %s in round %d.", selected_client, round_idx)

        elif aggregator == "trimmed_mean":
            new_flat = trimmed_mean_aggregate(flat_submissions, trim_ratio=0.2)

        elif aggregator == "fltrust":
            # Compute root gradient: run root data through global encoder + server model
            root_gradient = _compute_fltrust_root_gradient(state)
            ref_encoder = state.global_encoder or state.prev_global_encoder
            if ref_encoder is not None:
                # FLTrust trusts *updates*, not absolute weights — comparing raw
                # submitted weights to the root gradient is dominated by the
                # shared base model both start from. Aggregate deltas against
                # the reference encoder, then add back to reconstruct the
                # new global weights.
                ref_flat = state_dict_to_flat(ref_encoder)
                flat_deltas = {cid: w - ref_flat for cid, w in flat_submissions.items()}
                agg_delta = fltrust_aggregate(flat_deltas, root_gradient, sample_counts)
                new_flat = ref_flat + agg_delta
            else:
                # First round, no reference encoder yet — matches
                # fltrust_aggregate's own zero-root-gradient fallback.
                import numpy as np
                new_flat = np.stack(list(flat_submissions.values())).mean(axis=0)

        if new_flat is not None:
            new_global = flat_to_state_dict(new_flat, ref_sd, device=state.device)
            state.global_encoder = new_global
            logger.info(
                "Baseline '%s' round %d: global encoder updated from %d submissions.",
                aggregator, round_idx, len(submitted),
            )
        else:
            state.global_encoder = state.prev_global_encoder

        # Log round metrics (simpler than SNAS — no gate decisions)
        for cid in submitted:
            state.append_metric({
                "round": round_idx,
                "client_id": cid,
                "aggregator": aggregator,
                "included_in_aggregation": True,
                "krum_selected": (cid == selected_client) if selected_client is not None else None,
            })

        state.staleness_registry.update_after_round(set(submitted.keys()))
        state.current_round += 1


def _compute_fltrust_root_gradient(state: ServerState) -> "np.ndarray":
    """
    Compute a reference gradient for FLTrust from the server's root dataset.

    In the split-learning context, the server cannot run the client encoder
    directly (it doesn't hold the encoder weights on the data path). Instead,
    we use the current global encoder state to compute root activations, then
    backpropagate through the server model to obtain the root gradient w.r.t.
    encoder weights — treating the global encoder as the reference encoder.

    This approximation is the most faithful adaptation of FLTrust to the
    label-private split-learning setting, and is documented as such in the paper.
    """
    import numpy as np
    from sfl.common.models import ClientEncoder

    # global_encoder is cleared to None while a round's submission window is
    # open (app.py: "clear so clients get 404 during window") and this function
    # runs after that window closes — so state.global_encoder is None on every
    # single call. Falling back to prev_global_encoder (the last completed
    # round's encoder) instead of a hardcoded None check is what makes the root
    # gradient non-degenerate; without it root_norm is always ~0 and
    # fltrust_aggregate always takes its "falling back to uniform average" path.
    ref_encoder = state.global_encoder or state.prev_global_encoder
    if state.fltrust_root_data is None or ref_encoder is None:
        # No root data or no reference encoder yet (first round) — return zero
        # vector matching the encoder parameter count as a neutral reference
        param_count = sum(
            v.numel() for v in (ref_encoder or {}).values()
            if isinstance(v, __import__("torch").Tensor) and v.is_floating_point()
        )
        return np.zeros(param_count or 1, dtype=np.float32)

    import torch

    encoder = ClientEncoder(
        input_dim=int(state.config.get("input_dim", 265)),
        hidden_dim=int(state.config.get("encoder_hidden_dim", 64)),
    ).to(state.device)
    encoder.load_state_dict(ref_encoder)
    encoder.train()

    all_grads = []
    for cid, X_root in state.fltrust_root_data.items():
        y_root = state.fltrust_root_labels[cid]
        X_t = torch.tensor(X_root, dtype=torch.float32, device=state.device)
        y_t = torch.tensor(y_root, dtype=torch.float32, device=state.device).unsqueeze(1)

        activation = encoder(X_t)
        activation_detached = activation.detach().requires_grad_(True)
        logits = state.model(activation_detached)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y_t)

        # Get gradient w.r.t. activation, then compute encoder gradient
        loss.backward()
        act_grad = activation_detached.grad

        encoder.zero_grad()
        activation.backward(act_grad)

        # Walk state_dict() (not parameters()) so root_gradient's layout matches
        # state_dict_to_flat's — that includes BatchNorm's running_mean/running_var
        # buffers, which have no gradient and are padded with zeros here. Without
        # this, root_gradient (params-only) and the submitted deltas (full
        # state_dict) have different lengths and the cosine dot product in
        # fltrust_aggregate raises a shape mismatch.
        named_grads = {name: p.grad for name, p in encoder.named_parameters()}
        grads = []
        for key, tensor in encoder.state_dict().items():
            if not tensor.is_floating_point():
                continue
            grad = named_grads.get(key)
            if grad is not None:
                grads.append(grad.detach().cpu().float().numpy().ravel())
            else:
                grads.append(np.zeros(tensor.numel(), dtype=np.float32))
        if grads:
            all_grads.append(np.concatenate(grads))
        encoder.zero_grad()

    if not all_grads:
        return np.zeros(1, dtype=np.float32)

    root_gradient = np.stack(all_grads).mean(axis=0)
    return root_gradient


async def _process_round(state: ServerState, submitted: dict) -> None:
    """
    Core aggregation logic. Branches on aggregator_type:
      'snas'         -> full SNAS pipeline (gate + RobustAsyncFedAvg)
      'krum'         -> Krum selection (no staleness, no gate)
      'trimmed_mean' -> coordinate-wise trimmed mean (no staleness, no gate)
      'fltrust'      -> FLTrust trust-weighted aggregation (no staleness, no gate)
    """
    aggregator = state.config.get("aggregator_type", "snas")

    if aggregator != "snas":
        await _process_round_baseline(state, submitted, aggregator)
        return

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
            # Keep a handle so /admin/reset can cancel an in-flight round instead
            # of letting it wake up against freshly reset state.
            server_state.round_task = asyncio.create_task(_run_async_round(server_state))
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

    @app.post("/admin/reset", dependencies=[Depends(require_token)])
    async def admin_reset(request: ResetRequest) -> dict:
        """
        Reinitialise server state in place with optional config overrides.
        Used by automated multi-seed and sensitivity sweep scripts to avoid
        manual server restarts between experiment runs.
        """
        # A round opened by the previous experiment may still be sleeping out its
        # submission window. If it wakes after the reset it aggregates against
        # empty registries (KeyError on sample_counts) and tears down the
        # connections the next experiment is already using, so cancel it first.
        # Cancelled outside the lock because _run_async_round acquires it.
        task = getattr(server_state, "round_task", None)
        if task is not None and not task.done():
            logger.info("Reset requested while round %d in flight — cancelling it.",
                        server_state.current_round)
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass
        server_state.round_task = None

        async with server_state.lock:
            server_state.reset(request.config_overrides or None)
        return {
            "status": "reset",
            "aggregator_type": server_state.config.get("aggregator_type", "snas"),
            "seed": server_state.config.get("seed", 42),
        }

    return app


app = create_app()
