"""FastAPI backend for the Haiku Playground, deployed on Modal."""

from pathlib import Path

import modal

app = modal.App("haiku-playground")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("fastapi[standard]", "httpx", "nltk")
    .run_commands(
        "python -c \"import nltk; nltk.download('cmudict')\""
    )
    .add_local_dir("eval", "/root/eval")
    .add_local_dir("llm_judges", "/root/llm_judges")
)


@app.function(
    image=image,
    allow_concurrent_inputs=100,
)
@modal.asgi_app()
def serve_playground():
    from contextlib import asynccontextmanager

    import httpx
    import nltk
    from fastapi import FastAPI
    from fastapi.responses import FileResponse
    from pydantic import BaseModel

    from eval.shared import (
        MODEL_CHECKPOINTS,
        build_system_prompt,
        get_model_endpoint,
        query_model,
    )
    from llm_judges.base import MODAL_VOCABS
    from llm_judges.nlp import (
        count_syllables_for_word,
        score_haiku_structure,
        segment_haiku_lines,
    )

    class GenerateRequest(BaseModel):
        prompt: str
        step: str = "base"
        include_vocab: bool = True

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.cmudict = nltk.corpus.cmudict.dict()
        app.state.http_client = httpx.AsyncClient()
        yield
        await app.state.http_client.aclose()

    fastapi_app = FastAPI(title="Haiku Playground", lifespan=lifespan)

    @fastapi_app.post("/api/generate")
    async def generate(request: GenerateRequest):
        client = fastapi_app.state.http_client
        cmudict = fastapi_app.state.cmudict

        endpoint = get_model_endpoint(request.step)
        system_prompt = build_system_prompt(include_vocab=request.include_vocab)

        haiku = await query_model(
            client, endpoint, request.prompt, system_prompt=system_prompt
        )

        structure_score = score_haiku_structure(haiku, cmudict)

        # Compute per-line syllable counts
        import re

        lines = segment_haiku_lines(haiku)
        syllable_counts = []
        for line in lines:
            words = re.findall(r"[a-zA-Z]+", line)
            count = sum(count_syllables_for_word(w, cmudict) for w in words)
            syllable_counts.append(count)

        return {
            "haiku": haiku,
            "structure_score": structure_score,
            "syllable_counts": syllable_counts,
            "passed": structure_score >= 0.75,
        }

    @fastapi_app.get("/api/models")
    async def get_models():
        return MODEL_CHECKPOINTS

    @fastapi_app.get("/api/vocabs")
    async def get_vocabs():
        return MODAL_VOCABS

    @fastapi_app.get("/")
    async def index():
        html_path = Path("/root/eval/haiku_playground.html")
        return FileResponse(html_path, media_type="text/html")

    return fastapi_app
