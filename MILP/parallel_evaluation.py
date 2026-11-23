"""Independent CPU trajectory evaluations, with one process per pending trial."""
from __future__ import annotations

import multiprocessing as mp
from multiprocessing.connection import wait
import os
import time


def worker_count():
    value = os.environ.get("METRO_CANDIDATE_WORKERS", "1")
    count = int(value)
    if count < 0:
        raise ValueError("METRO_CANDIDATE_WORKERS must be 0 (all) or positive")
    return count


def _evaluate(model, selection, sender):
    try:
        value = model.evaluate(selection)
        sender.send(("complete", value, time.perf_counter()))
    except BaseException as exc:
        from MILP.operation_model import EvaluationTimeout
        sender.send(("timeout" if isinstance(exc, EvaluationTimeout) else "error",
                     type(exc).__name__, time.perf_counter()))
    finally:
        sender.close()


def evaluate_batch(model, selections):
    """Only fully returned, independently checked trajectories enter the result."""
    keys = list(dict.fromkeys(tuple(int(i) for i in selection) for selection in selections))
    count = worker_count()
    started = time.perf_counter()
    stats = dict(requested=len(keys), launched=0, completed=0, timed_out=0, peak_workers=0,
                 mode="serial" if count == 1 else "fork_processes", wall_seconds=0.)
    completed = {}
    if not keys:
        return completed, stats
    model.ensure_reference()
    if count == 1:
        from MILP.operation_model import EvaluationTimeout
        for key in keys:
            try:
                completed[key] = model.evaluate(key)
                stats["completed"] += 1
            except EvaluationTimeout:
                break
        stats.update(timed_out=len(keys)-len(completed), wall_seconds=time.perf_counter()-started)
        return completed, stats
    if "fork" not in mp.get_all_start_methods():
        raise RuntimeError("Parallel physical evaluations require a POSIX fork-capable CPU host")
    context = mp.get_context("fork")
    running = {}
    next_index = 0
    maximum = len(keys) if count == 0 else count
    try:
        while next_index < len(keys) or running:
            while next_index < len(keys) and len(running) < maximum and time.perf_counter() < model.deadline:
                key = keys[next_index]
                next_index += 1
                receiver, sender = context.Pipe(duplex=False)
                child = context.Process(target=_evaluate, args=(model, key, sender), daemon=False)
                try:
                    child.start()
                except BaseException:
                    receiver.close()
                    sender.close()
                    raise
                sender.close()
                running[receiver] = (child, key)
                stats["launched"] += 1
                stats["peak_workers"] = max(stats["peak_workers"], len(running))
            remaining = model.deadline-time.perf_counter()
            if remaining <= 0:
                break
            if not running:
                break
            for receiver in wait(list(running), timeout=min(.1, remaining)):
                child, key = running.pop(receiver)
                try:
                    kind, payload, finished = receiver.recv()
                    if kind == "complete" and finished <= model.deadline and time.perf_counter() <= model.deadline:
                        if payload.get("evaluation_catalog") != model.catalog_fingerprint or payload.get("evaluation_selection") != key:
                            raise ValueError("Parallel trajectory belongs to another state or candidate catalog")
                        completed[key] = payload
                        stats["completed"] += 1
                    elif kind == "error":
                        raise RuntimeError(f"Physical evaluation worker failed: {payload}")
                except EOFError as exc:
                    raise RuntimeError("Physical evaluation worker exited without a complete result") from exc
                finally:
                    receiver.close()
                    child.join(timeout=.2)
                    if child.is_alive():
                        child.terminate()
                        child.join()
    finally:
        for receiver, (child, _) in running.items():
            receiver.close()
            if child.is_alive():
                child.terminate()
            child.join(timeout=.5)
            if child.is_alive():
                child.kill()
                child.join()
    stats.update(timed_out=len(keys)-len(completed), wall_seconds=time.perf_counter()-started)
    return {key: completed[key] for key in keys if key in completed}, stats
