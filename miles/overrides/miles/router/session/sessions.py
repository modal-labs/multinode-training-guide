import json
import logging
import time
from typing import TYPE_CHECKING

from fastapi import Request
from fastapi.responses import JSONResponse, Response
from transformers import AutoTokenizer

from miles.router.session.naive_trajectory import NaiveTrajectoryManager
from miles.router.session.session_types import GetSessionResponse, SessionRecord

if TYPE_CHECKING:
    from miles.router.router import MilesRouter

logger = logging.getLogger(__name__)


def _ensure_token_ids(choice: dict, tokenizer) -> None:
    logprobs = choice.get("logprobs", {})
    content = logprobs.get("content")
    if not content:
        raise RuntimeError("logprobs must be in choice")

    message = choice.get("message", {})
    message_content = message.get("content") or ""
    encoded_message = tokenizer.encode(message_content, add_special_tokens=False)
    if len(encoded_message) == len(content):
        for item, token_id in zip(content, encoded_message, strict=True):
            item.setdefault("token_id", int(token_id))
        return

    for item in content:
        if "token_id" in item:
            continue
        token = item.get("token")
        if token is None:
            logger.error(
                "Failed to recover token ids from chat completion. choice keys=%s content keys=%s message_len=%s",
                sorted(choice.keys()),
                sorted(item.keys()),
                len(message_content),
            )
            raise RuntimeError("token_id must be in choice's logprobs content item")
        token_id = tokenizer.convert_tokens_to_ids(token)
        if token_id is None:
            raise RuntimeError("token_id must be in choice's logprobs content item")
        item["token_id"] = int(token_id)


def _ensure_prompt_token_ids(choice: dict, request_body: dict, tokenizer) -> None:
    if choice.get("prompt_token_ids") is not None:
        return

    input_ids = request_body.get("input_ids")
    if input_ids is not None:
        choice["prompt_token_ids"] = input_ids
        return

    messages = request_body.get("messages")
    if messages is None:
        raise RuntimeError("prompt_token_ids missing in chat response")

    choice["prompt_token_ids"] = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)


def setup_session_routes(app, router: "MilesRouter"):
    hf_checkpoint = getattr(router.args, "hf_checkpoint", None)
    if not hf_checkpoint:
        if getattr(router, "verbose", False):
            logger.info("[miles-router] Skipping session routes (hf_checkpoint not set).")
        return

    tokenizer = AutoTokenizer.from_pretrained(hf_checkpoint, trust_remote_code=True)
    manager = NaiveTrajectoryManager(router.args, tokenizer)

    @app.post("/sessions")
    async def create_session():
        session_id = manager.create_session()
        return {"session_id": session_id}

    @app.get("/sessions/{session_id}")
    async def get_session(session_id: str):
        records = manager.get_session_records_by_id(session_id)
        if records is None:
            return JSONResponse(status_code=404, content={"error": "session not found"})
        return GetSessionResponse(
            session_id=session_id,
            records=records,
        )

    @app.delete("/sessions/{session_id}")
    async def delete_session(session_id: str):
        deleted = manager.delete_session_by_id(session_id)
        if deleted is None:
            return JSONResponse(status_code=404, content={"error": "session not found"})
        return Response(status_code=204)

    @app.post("/sessions/{session_id}/v1/chat/completions")
    async def chat_completions(request: Request, session_id: str):
        body = await request.body()
        request_body = json.loads(body) if body else {}

        request_body.setdefault("logprobs", True)
        request_body.setdefault("return_prompt_token_ids", True)
        body = json.dumps(request_body).encode()

        result = await router._do_proxy(request, "v1/chat/completions", body=body)
        if result["status_code"] != 200:
            return router._build_proxy_response(result)

        response = json.loads(result["response_body"])
        choice = response.get("choices", [{}])[0]
        _ensure_token_ids(choice, tokenizer)
        _ensure_prompt_token_ids(choice, request_body, tokenizer)

        record = SessionRecord(
            timestamp=time.time(),
            method=request.method,
            path="/v1/chat/completions",
            status_code=result["status_code"],
            request=request_body,
            response=response,
        )
        appended = manager.append_session_record(session_id, record)
        if appended is None:
            return JSONResponse(status_code=404, content={"error": "session not found"})
        result["response_body"] = json.dumps(response).encode()
        result["headers"] = {
            key: value
            for key, value in result["headers"].items()
            if key.lower() not in {"content-length", "transfer-encoding"}
        }
        return router._build_proxy_response(result)

    @app.api_route("/sessions/{session_id}/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
    async def session_proxy(request: Request, session_id: str, path: str):
        result = await router._do_proxy(request, path)
        return router._build_proxy_response(result)
