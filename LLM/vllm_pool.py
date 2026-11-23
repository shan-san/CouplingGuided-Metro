"""One local OpenAI endpoint for independent, identical vLLM GPU replicas.

Run with one gateway process so all experiment clients share the same counters.
Generation requests are never retried; health probes may recover a replica.
"""
from __future__ import annotations

import argparse
import asyncio
from contextlib import suppress
from dataclasses import dataclass
import json
import logging
import math
import os
from pathlib import Path
import time
from urllib.parse import urlsplit

import httpx


logger = logging.getLogger(__name__)


@dataclass
class Replica:
    url: str
    healthy: bool = False
    inflight: int = 0
    assigned: int = 0
    completed: int = 0
    failed: int = 0
    cancelled: int = 0
    elapsed_s: float = 0.
    last_error: str | None = None
    models: dict | None = None


class PoolUnavailable(Exception):
    pass


def backend_url(value):
    url = urlsplit(value)
    if (url.scheme not in ("http", "https") or not url.hostname or url.username or url.password
            or url.query or url.fragment or url.path.rstrip("/") != "/v1"):
        raise ValueError("Backend URLs must end in /v1 without credentials, queries or fragments")
    if url.port is not None and not 0 < url.port < 65536:
        raise ValueError("Invalid backend port")
    return value.rstrip("/")


class VLLMPool:
    def __init__(self, backends, *, model="metro-slm-v9", adapter=None, per_backend=4,
                 timeout_s=60., queue_timeout_s=15., health_interval_s=5., max_waiting=64,
                 startup_timeout_s=0., client=None):
        urls = [backend_url(value) for value in backends]
        if not urls or len(urls) != len(set(urls)):
            raise ValueError("Provide distinct vLLM backends")
        if type(per_backend) is not int or per_backend < 1 or type(max_waiting) is not int or max_waiting < 1:
            raise ValueError("Replica concurrency and waiting capacity must be positive integers")
        if any(not math.isfinite(v) or v <= 0 for v in (timeout_s, queue_timeout_s, health_interval_s)):
            raise ValueError("Timeouts and health interval must be finite and positive")
        if not math.isfinite(startup_timeout_s) or startup_timeout_s < 0:
            raise ValueError("Startup waiting time must be finite and nonnegative")
        self.replicas = [Replica(url) for url in urls]
        self.model = model
        self.adapter = str(Path(adapter).expanduser().resolve()) if adapter else None
        self.per_backend, self.max_waiting = per_backend, max_waiting
        self.timeout_s, self.queue_timeout_s, self.health_interval_s = timeout_s, queue_timeout_s, health_interval_s
        self.startup_timeout_s = startup_timeout_s
        self.client = client
        self._owns_client = client is None
        self._condition = asyncio.Condition()
        self._cursor = self.waiting = 0
        self._identity = None
        self._monitor = None

    def _model_identity(self, document):
        if not isinstance(document, dict) or not isinstance(document.get("data"), list):
            raise ValueError("Invalid replica model metadata")
        selected = next((m for m in document["data"] if isinstance(m, dict) and m.get("id") == self.model), None)
        if not selected or not isinstance(selected.get("parent"), str) or not selected["parent"]:
            raise ValueError("Expected a served LoRA adapter, not the base model")
        root = selected.get("root")
        if root is not None and not isinstance(root, str):
            raise ValueError("Invalid replica adapter root")
        root = str(Path(root).resolve()) if root else None
        if self.adapter and root != self.adapter:
            raise ValueError("Replica serves a different adapter path")
        return self.model, selected["parent"], root

    async def probe(self):
        async def check(replica):
            try:
                health_url = replica.url.removesuffix("/v1") + "/health"
                health = await asyncio.wait_for(self.client.get(health_url, timeout=5.), 5.)
                health.raise_for_status()
                response = await asyncio.wait_for(self.client.get(replica.url + "/models", timeout=5.), 5.)
                response.raise_for_status()
                models = response.json()
                identity = self._model_identity(models)
                if self._identity is None:
                    self._identity = identity
                if identity != self._identity:
                    raise ValueError("Replica model identity differs from the pool")
                async with self._condition:
                    replica.healthy, replica.last_error, replica.models = True, None, models
                    self._condition.notify_all()
            except (httpx.HTTPError, TimeoutError, ValueError, KeyError, TypeError) as exc:
                async with self._condition:
                    replica.healthy, replica.last_error = False, type(exc).__name__
                    self._condition.notify_all()
        await asyncio.gather(*(check(replica) for replica in self.replicas))

    async def start(self):
        if self.client is None:
            self.client = httpx.AsyncClient(trust_env=False, follow_redirects=False,
                headers={"Authorization": "Bearer " + os.environ.get("METRO_VLLM_BACKEND_KEY", "EMPTY")},
                limits=httpx.Limits(max_connections=len(self.replicas)*(self.per_backend+2)),
                timeout=self.timeout_s)
        deadline = time.monotonic() + self.startup_timeout_s
        try:
            while True:
                await self.probe()
                if all(replica.healthy for replica in self.replicas):
                    break
                if any(r.last_error == "ValueError" for r in self.replicas) or time.monotonic() >= deadline:
                    raise RuntimeError("All configured replicas must be ready and serve the same adapter before startup")
                logger.info("waiting_for_replicas healthy=%s total=%s unavailable=%s",
                    sum(r.healthy for r in self.replicas), len(self.replicas),
                    [(i, r.last_error) for i, r in enumerate(self.replicas) if not r.healthy])
                await asyncio.sleep(min(2., max(0., deadline-time.monotonic())))
        except BaseException:
            await self.close()
            raise
        self._monitor = asyncio.create_task(self._health_loop())

    async def _health_loop(self):
        while True:
            await asyncio.sleep(self.health_interval_s)
            await self.probe()

    async def close(self):
        if self._monitor is not None:
            self._monitor.cancel()
            with suppress(asyncio.CancelledError):
                await self._monitor
            self._monitor = None
        if self._owns_client and self.client is not None:
            await self.client.aclose()

    async def _acquire(self):
        deadline = time.monotonic() + self.queue_timeout_s
        async with self._condition:
            if self.waiting >= self.max_waiting:
                raise PoolUnavailable("Inference waiting queue is full")
            self.waiting += 1
            try:
                while True:
                    healthy = [i for i, r in enumerate(self.replicas) if r.healthy]
                    if not healthy:
                        raise PoolUnavailable("No healthy inference replica")
                    available = [i for i in healthy if self.replicas[i].inflight < self.per_backend]
                    if available:
                        index = min(available, key=lambda i: (self.replicas[i].inflight,
                            (i-self._cursor) % len(self.replicas)))
                        replica = self.replicas[index]
                        replica.inflight += 1
                        replica.assigned += 1
                        self._cursor = (index+1) % len(self.replicas)
                        return index, replica
                    remaining = deadline-time.monotonic()
                    if remaining <= 0:
                        raise PoolUnavailable("Inference queue deadline exceeded")
                    try:
                        await asyncio.wait_for(self._condition.wait(), remaining)
                    except TimeoutError as exc:
                        raise PoolUnavailable("Inference queue deadline exceeded") from exc
            finally:
                self.waiting -= 1

    async def generate(self, body):
        started = time.monotonic()
        index = replica = None
        queued_s = 0.
        status = 499
        try:
            index, replica = await self._acquire()
            queued_s = time.monotonic()-started
            # Full response body reception is inside the same wall-clock limit.
            response = await asyncio.wait_for(self.client.post(replica.url + "/chat/completions",
                content=body, headers={"Content-Type": "application/json"}, timeout=self.timeout_s), self.timeout_s)
            status = response.status_code
            if response.is_error:
                replica.failed += 1
            else:
                replica.completed += 1
            if status >= 500:
                replica.healthy, replica.last_error = False, "upstream_http_5xx"
            return status, response.content, {"content-type": response.headers.get("content-type", "application/json"),
                "x-metro-replica": str(index)}
        except asyncio.CancelledError:
            if replica is not None:
                replica.cancelled += 1
            raise
        except (httpx.HTTPError, TimeoutError) as exc:
            status = 504 if isinstance(exc, (TimeoutError, httpx.TimeoutException)) else 502
            if replica is not None:
                replica.failed += 1
                replica.healthy, replica.last_error = False, type(exc).__name__
            return status, json.dumps({"error": {"message": "vLLM request failed; generation was not retried",
                "type": type(exc).__name__}}).encode(), {"content-type": "application/json"}
        finally:
            if replica is not None:
                elapsed = time.monotonic()-started
                async with self._condition:
                    replica.inflight -= 1
                    replica.elapsed_s += elapsed
                    self._condition.notify_all()
                logger.info("inference replica=%s status=%s queue_s=%.3f elapsed_s=%.3f",
                            index, status, queued_s, elapsed)

    def snapshot(self):
        return dict(model=self.model, healthy_replicas=sum(r.healthy for r in self.replicas),
            replicas=len(self.replicas), waiting=self.waiting, policy="least_inflight_then_round_robin",
            per_backend=self.per_backend, generation_retries=0,
            backends=[dict(index=i, url=r.url, healthy=r.healthy, inflight=r.inflight,
                assigned=r.assigned, completed=r.completed, failed=r.failed, cancelled=r.cancelled,
                elapsed_s=r.elapsed_s, last_error=r.last_error) for i, r in enumerate(self.replicas)])


class PoolApplication:
    """Small ASGI app; uvicorn supplies HTTP parsing, sockets and shutdown."""
    def __init__(self, pool, max_body_bytes=2*1024*1024):
        self.pool, self.max_body_bytes = pool, max_body_bytes

    async def __call__(self, scope, receive, send):
        if scope["type"] == "lifespan":
            while True:
                event = await receive()
                if event["type"] == "lifespan.startup":
                    try:
                        await self.pool.start()
                    except Exception as exc:
                        await send(dict(type="lifespan.startup.failed", message=str(exc)))
                        return
                    await send(dict(type="lifespan.startup.complete"))
                elif event["type"] == "lifespan.shutdown":
                    await self.pool.close()
                    await send(dict(type="lifespan.shutdown.complete"))
                    return
        if scope["type"] != "http":
            return
        method, path = scope["method"], scope["path"]
        if method == "GET" and path in ("/health", "/pool/status"):
            return await self._json(send, 200 if any(r.healthy for r in self.pool.replicas) else 503, self.pool.snapshot())
        if method == "GET" and path == "/v1/models":
            replica = next((r for r in self.pool.replicas if r.healthy), None)
            return await self._json(send, 200, replica.models) if replica else await self._json(
                send, 503, {"error": "No healthy inference replica"})
        if method != "POST" or path != "/v1/chat/completions":
            return await self._json(send, 404, {"error": "Unknown inference endpoint"})
        body = bytearray()
        while True:
            event = await receive()
            if event["type"] == "http.disconnect":
                return
            body.extend(event.get("body", b""))
            if len(body) > self.max_body_bytes:
                return await self._json(send, 413, {"error": "Inference request exceeds the body limit"})
            if not event.get("more_body", False):
                break
        try:
            payload = json.loads(body)
            if not isinstance(payload, dict) or payload.get("model") != self.pool.model or payload.get("stream", False):
                raise ValueError()
        except (ValueError, UnicodeDecodeError):
            return await self._json(send, 400, {"error": "Expected a non-streaming request for the configured adapter"})
        generation = asyncio.create_task(self.pool.generate(bytes(body)))
        disconnected = asyncio.create_task(self._disconnect(receive))
        try:
            done, _ = await asyncio.wait((generation, disconnected), return_when=asyncio.FIRST_COMPLETED)
            if disconnected in done:
                return
            try:
                status, content, headers = generation.result()
            except PoolUnavailable as exc:
                return await self._json(send, 503, {"error": str(exc)})
            await send(dict(type="http.response.start", status=status,
                headers=[(key.encode(), value.encode()) for key, value in headers.items()]))
            await send(dict(type="http.response.body", body=content))
        finally:
            for task in (generation, disconnected):
                if not task.done():
                    task.cancel()
            await asyncio.gather(generation, disconnected, return_exceptions=True)

    @staticmethod
    async def _disconnect(receive):
        while (await receive())["type"] != "http.disconnect":
            pass

    @staticmethod
    async def _json(send, status, value):
        await send(dict(type="http.response.start", status=status,
            headers=[(b"content-type", b"application/json")]))
        await send(dict(type="http.response.body", body=json.dumps(value).encode()))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backends", nargs="+", required=True, help="Replica URLs, each ending in /v1")
    parser.add_argument("--adapter", required=True, help="The same trained adapter path on every replica")
    parser.add_argument("--model", default="metro-slm-v9")
    parser.add_argument("--host", choices=("127.0.0.1", "::1"), default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--per-backend", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=60.)
    parser.add_argument("--queue-timeout", type=float, default=15.)
    parser.add_argument("--startup-timeout", type=float, default=900., help="Wait for every replica to load before serving")
    args = parser.parse_args(argv)
    if not 0 < args.port < 65536:
        parser.error("Invalid gateway port")
    pool = VLLMPool(args.backends, model=args.model, adapter=args.adapter,
        per_backend=args.per_backend, timeout_s=args.timeout, queue_timeout_s=args.queue_timeout,
        startup_timeout_s=args.startup_timeout)
    import uvicorn
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    uvicorn.run(PoolApplication(pool), host=args.host, port=args.port, workers=1, access_log=False)


if __name__ == "__main__":
    main()
