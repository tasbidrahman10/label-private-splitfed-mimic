from __future__ import annotations

import uuid

import torch

from sfl.common.serialization import payload_to_tensor, tensor_to_payload
from sfl.server.state import OptionBContext, ServerState


@torch.no_grad()
def predict_from_activation(state: ServerState, activation_payload: list) -> dict:
    activation = payload_to_tensor(activation_payload, state.device)
    state.model.eval()
    logits = state.model(activation)
    return {"logits": tensor_to_payload(logits)}


def option_b_forward(
    state: ServerState,
    client_id: int,
    activation_payload: list,
    round_idx: int | None = None,
    batch_idx: int | None = None,
) -> dict:
    activation = payload_to_tensor(activation_payload, state.device).detach()
    activation.requires_grad_(True)

    state.model.train()
    logits = state.model(activation)
    context_id = str(uuid.uuid4())
    state.option_b_contexts[context_id] = OptionBContext(
        client_id=client_id,
        activation=activation,
        logits=logits,
        round_idx=round_idx,
        batch_idx=batch_idx,
    )
    return {
        "context_id": context_id,
        "logits": tensor_to_payload(logits.detach()),
    }


def option_b_backward(
    state: ServerState,
    client_id: int,
    context_id: str,
    logit_gradient_payload: list,
) -> dict:
    context = state.option_b_contexts.pop(context_id)
    if context.client_id != client_id:
        raise ValueError("Option B context does not belong to this client")

    logit_grad = payload_to_tensor(logit_gradient_payload, state.device)
    state.optimizer.zero_grad()
    context.logits.backward(logit_grad)
    activation_grad = context.activation.grad.detach().clone()
    state.optimizer.step()

    state.append_metric(
        {
            "mode": "option_b",
            "client_id": client_id,
            "round": context.round_idx,
            "batch": context.batch_idx,
            "batch_size": int(logit_grad.shape[0]),
        }
    )
    return {"activation_gradient": tensor_to_payload(activation_grad)}
