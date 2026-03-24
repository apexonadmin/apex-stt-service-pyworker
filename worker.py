"""PyWorker configuration for VastAI Serverless WhisperX transcription.

This module configures the VastAI PyWorker to proxy requests through
nginx load balancer on localhost:8080, which distributes across 6
model servers on ports 5000-5005.

Architecture:
    VastAI Router -> PyWorker (:5006) -> nginx (:8080) -> 6x Model Servers (:5000-5005)
                                                           -> WhisperX transcribe
                                                           -> Pyannote diarize

The PyWorker handles:
- Request validation and routing
- Workload cost calculation for autoscaling
- Readiness detection via log parsing
- Metrics reporting to VastAI

NOTE: This file is the canonical copy for VastAI's PYWORKER_REPO.
      A reference copy exists in py-ts-serverless/worker.py — keep in sync manually.
"""

import os

from vastai import (
    BenchmarkConfig,
    HandlerConfig,
    LogActionConfig,
    Worker,
    WorkerConfig,
)

# Model server configuration (runs on same container)
MODEL_SERVER_HOST = os.getenv("MODEL_SERVER_HOST", "localhost")
MODEL_SERVER_PORT = int(os.getenv("MODEL_SERVER_PORT", "8080"))
# MODEL_LOG: VastAI's start_server.sh sets MODEL_LOG env var
MODEL_LOG = os.getenv("MODEL_LOG", "/tmp/model_server.log")

# Benchmark audio URL (public sample for warmup/capacity estimation)
# Default: NASA Apollo 11 sample (public domain, reliable archive.org CDN)
# Override via BENCHMARK_AUDIO_URL env var for custom benchmark audio
BENCHMARK_AUDIO_URL = os.getenv(
    "BENCHMARK_AUDIO_URL",
    "https://storage.googleapis.com/cloud-samples-tests/speech/brooklyn.flac",
)


def workload_calculator(payload: dict) -> float:
    """Calculate workload cost for VastAI autoscaling.

    Formula:
    - Without duration: flat 100.0 base cost
    - With duration_seconds: max(50.0, duration * 100/60)
      e.g. 10s→50, 30s→50, 60s→100, 120s→200
    - Diarization: +50% multiplier on base cost
    - Invalid/missing duration: falls back to flat 100.0
    """
    duration = payload.get("duration_seconds")
    if duration and isinstance(duration, (int, float)) and duration > 0:
        base_cost = max(50.0, duration * (100.0 / 60.0))
    else:
        base_cost = 100.0
    if payload.get("diarize", False):
        base_cost *= 1.5
    return base_cost


def benchmark_generator() -> dict:
    """Generate a benchmark payload for capacity estimation."""
    return {
        "audio_url": BENCHMARK_AUDIO_URL,
        "language": "en",
        "align": True,
        "diarize": False,
    }


# Handler for /transcribe endpoint
transcribe_handler = HandlerConfig(
    route="/transcribe",
    workload_calculator=workload_calculator,
    allow_parallel_requests=True,
    max_queue_time=120.0,
    # Benchmark config: runs at startup to calibrate throughput
    benchmark_config=BenchmarkConfig(
        generator=benchmark_generator,
        runs=4,
        concurrency=4,
    ),
)


# Handler for /health endpoint (free health checks)
health_handler = HandlerConfig(
    route="/health",
    workload_calculator=lambda payload: 0.0,
    allow_parallel_requests=True,
    max_queue_time=5.0,
)


# Log-based readiness detection (patterns must be lists)
log_actions = LogActionConfig(
    on_load=["Model loaded"],
    on_error=[
        "CUDA out of memory",
        "RuntimeError:",
        "Traceback (most recent call last):",
        "ERROR:    Uvicorn running on",
        "ERROR:    [Errno",
        "torch.cuda.OutOfMemoryError",
    ],
    on_info=[
        "Starting model server",
        "Models downloaded successfully",
        "Pre-baked models detected",
        "Waiting for",
        "All ",
        "Starting nginx",
        "Starting PyWorker",
        "Preprocessing complete",
    ],
)


# Main worker configuration
worker_config = WorkerConfig(
    model_server_url=f"http://{MODEL_SERVER_HOST}",
    model_server_port=MODEL_SERVER_PORT,
    model_log_file=MODEL_LOG,
    handlers=[transcribe_handler, health_handler],
    log_action_config=log_actions,
    # Health check path for SDK to confirm model readiness after benchmark
    # SDK prepends model_server_url, so use relative path only
    model_healthcheck_url="/health",
)


if __name__ == "__main__":
    Worker(worker_config).run()
