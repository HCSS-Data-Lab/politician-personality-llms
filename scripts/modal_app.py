"""Optional authenticated Modal deployment for the published open-weight panel.

Create a Modal secret named politician-personality-llms containing your own
HF_TOKEN and VLLM_API_KEY, then deploy this file. Accept model access terms in
your own Hugging Face account. Deployment and live calls incur your own charges.
"""
from __future__ import annotations

import os
import subprocess
import modal

APP_NAME = "politician-personality-llms"
VLLM_VERSION = "0.21.0"
CUDA_IMAGE = "nvidia/cuda:12.9.0-devel-ubuntu22.04"
PORT = 8000
MODELS = {
    "qwen-8b": {"hf_id": "Qwen/Qwen3-8B", "revision": "b968826d9c46dd6066d109eabc6255188de91218", "gpu": "A100-40GB", "dtype": "float16"},
    "qwen-32b": {"hf_id": "Qwen/Qwen3-32B", "revision": "9216db5781bf21249d130ec9da846c4624c16137", "gpu": "A100-80GB", "dtype": "float16"},
    "gemma-12b": {"hf_id": "google/gemma-3-12b-it", "revision": "96b6f1eccf38110c56df3a15bffe176da04bfd80", "gpu": "A100-40GB", "dtype": "bfloat16"},
    "gemma-27b": {"hf_id": "google/gemma-3-27b-it", "revision": "005ad3404e59d6023443cb575daa05336842228a", "gpu": "A100-80GB", "dtype": "bfloat16"},
}

# This compatibility patch was used with the pinned serving stack: some router
# entries lack .path. Fail explicitly if the installed source no longer matches.
ROUTING_PATCH = """python - <<'PYCODE'
import prometheus_fastapi_instrumentator.routing as routing
from pathlib import Path
path = Path(routing.__file__)
source = path.read_text()
old = 'route_name = route.path'
assert source.count(old) == 2, 'Unexpected instrumentation source; revalidate the serving environment'
path.write_text(source.replace(old, 'route_name = getattr(route, "path", "")'))
PYCODE"""

image = (modal.Image.from_registry(CUDA_IMAGE, add_python="3.12")
         .entrypoint([])
         .pip_install(f"vllm=={VLLM_VERSION}", "huggingface_hub[hf_xet]")
         .run_commands(ROUTING_PATCH)
         .env({"HF_XET_HIGH_PERFORMANCE": "1"}))
secret = modal.Secret.from_name(APP_NAME, required_keys=["HF_TOKEN", "VLLM_API_KEY"])
volumes = {
    "/root/.cache/huggingface": modal.Volume.from_name(APP_NAME + "-weights", create_if_missing=True),
    "/root/.cache/vllm": modal.Volume.from_name(APP_NAME + "-compile", create_if_missing=True),
}
app = modal.App(APP_NAME)


def server_command(model: str, api_key: str) -> list[str]:
    if not api_key:
        raise ValueError("VLLM_API_KEY is required; unauthenticated endpoints are not enabled")
    spec = MODELS[model]
    return ["vllm", "serve", spec["hf_id"], "--revision", spec["revision"],
            "--served-model-name", spec["hf_id"], "--host", "0.0.0.0",
            "--port", str(PORT), "--dtype", spec["dtype"],
            "--max-model-len", "20480", "--api-key", api_key]


def serve(model: str):
    subprocess.Popen(server_command(model, os.environ.get("VLLM_API_KEY", "")))


def settings(model: str):
    return {"image": image, "gpu": MODELS[model]["gpu"], "volumes": volumes,
            "secrets": [secret], "scaledown_window": 600, "timeout": 14400}


@app.function(**settings("qwen-8b"))
@modal.concurrent(max_inputs=64)
@modal.web_server(port=PORT, startup_timeout=1800)
def serve_qwen_8b():
    serve("qwen-8b")


@app.function(**settings("qwen-32b"))
@modal.concurrent(max_inputs=64)
@modal.web_server(port=PORT, startup_timeout=1800)
def serve_qwen_32b():
    serve("qwen-32b")


@app.function(**settings("gemma-12b"))
@modal.concurrent(max_inputs=64)
@modal.web_server(port=PORT, startup_timeout=1800)
def serve_gemma_12b():
    serve("gemma-12b")


@app.function(**settings("gemma-27b"))
@modal.concurrent(max_inputs=64)
@modal.web_server(port=PORT, startup_timeout=1800)
def serve_gemma_27b():
    serve("gemma-27b")


@app.local_entrypoint()
def endpoints():
    for key in MODELS:
        function = modal.Function.from_name(APP_NAME, "serve_" + key.replace("-", "_"))
        url = function.get_web_url()
        if not url:
            raise RuntimeError("Deploy the app before resolving endpoints")
        print(key, url.rstrip("/") + "/v1")
