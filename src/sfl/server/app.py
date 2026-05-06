from __future__ import annotations

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


def create_app(config_path: str = "configs/server.yaml") -> FastAPI:
    config = load_yaml(config_path)
    server_state = ServerState(config)
    app = FastAPI(title="MIMIC SplitFed Server", version="0.1.0")

    def require_token(authorization: str | None = Header(default=None)) -> None:
        expected = server_state.expected_token()
        if not expected:
            return
        if authorization != f"Bearer {expected}":
            raise HTTPException(status_code=401, detail="Invalid or missing bearer token")

    @app.get("/health")
    def health() -> dict:
        return {
            "status": "ok",
            "device": str(server_state.device),
            "registered_clients": len(server_state.clients),
        }

    @app.post("/register_client", dependencies=[Depends(require_token)])
    def register_client(request: RegisterRequest) -> dict:
        with server_state.lock:
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
    def predict(request: PredictRequest) -> dict:
        with server_state.lock:
            return predict_from_activation(server_state, request.activation)

    @app.post("/option-b/forward", dependencies=[Depends(require_token)])
    def option_b_forward_endpoint(request: OptionBForwardRequest) -> dict:
        with server_state.lock:
            return option_b_forward(
                server_state,
                client_id=request.client_id,
                activation_payload=request.activation,
                round_idx=request.round,
                batch_idx=request.batch,
            )

    @app.post("/option-b/backward", dependencies=[Depends(require_token)])
    def option_b_backward_endpoint(request: OptionBBackwardRequest) -> dict:
        with server_state.lock:
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

    @app.post("/fedavg/submit_encoder", dependencies=[Depends(require_token)])
    def submit_encoder(request: EncoderSubmitRequest) -> dict:
        with server_state.lock:
            state_dict = base64_to_state_dict(request.state_dict, server_state.device)
            status = server_state.submit_encoder(request.client_id, state_dict, request.sample_count)
        return {"status": "submitted", **status}

    @app.get("/fedavg/global_encoder", dependencies=[Depends(require_token)])
    def get_global_encoder() -> dict:
        with server_state.lock:
            if server_state.global_encoder is None:
                raise HTTPException(status_code=404, detail="No global encoder available yet")
            payload = state_dict_to_base64(server_state.global_encoder)
        return {"state_dict": payload}

    @app.get("/metrics", dependencies=[Depends(require_token)])
    def metrics() -> dict:
        return {"metrics": server_state.metrics[-200:]}

    @app.post("/checkpoint/server", dependencies=[Depends(require_token)])
    def save_server_checkpoint(request: SaveServerRequest) -> dict:
        with server_state.lock:
            path = server_state.save_checkpoint(request.filename)
        return {"status": "saved", "path": str(path)}

    return app


app = create_app()
