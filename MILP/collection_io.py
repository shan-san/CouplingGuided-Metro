"""Transactional collection files and an isolated, tokenizer-only worker."""
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import pickle
import select
import subprocess
import sys
import tempfile


STREAMS = ("solver_records.jsonl", "teacher_states.jsonl", "sft.jsonl")


def repair_failure_tail(path):
    """Drop only an interrupted final record; interior corruption is an error."""
    path = Path(path)
    if not path.exists():
        return 0
    count, offset = 0, 0
    with path.open("r+b") as stream:
        for line in stream:
            if not line.endswith(b"\n"):
                stream.truncate(offset)
                break
            json.loads(line)
            offset += len(line)
            count += 1
    return count


def atomic_json(path, value):
    atomic_bytes(path, (json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2)+"\n").encode())


def atomic_bytes(path, value):
    path = Path(path)
    fd, temporary = tempfile.mkstemp(prefix="."+path.name, dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


@contextmanager
def collection_lock(output):
    # Linux and macOS release this lock automatically after a killed collector.
    import fcntl
    with (Path(output)/".collection.lock").open("a+") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another collector owns this output directory") from exc
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


class CollectionJournal:
    """Commit the world and byte offsets together; resume discards torn tails.

    Checkpoints contain Python objects and must only be loaded from the user's
    own trusted collection directory, with matching source/config fingerprints.
    """
    def __init__(self, output, identity, *, resume=False, overwrite=False):
        self.output = Path(output)
        self.identity = identity
        self.path = self.output/"checkpoint.pkl"
        self.streams = {}
        existing = [self.output/name for name in (*STREAMS, "run_manifest.json", "checkpoint.pkl", "summary.json", "failures.jsonl")
                    if (self.output/name).exists()]
        if resume and overwrite:
            raise ValueError("Choose resume or overwrite, not both")
        if existing and not (resume or overwrite):
            raise FileExistsError("Output contains a collection; use --resume or a new output directory")
        if overwrite:
            for name in (*STREAMS, "run_manifest.json", "checkpoint.pkl", "summary.json", "progress.json", "rag.json", "rag.partial.json", "failures.jsonl"):
                (self.output/name).unlink(missing_ok=True)
        self.restored = None
        if resume:
            manifest = json.loads((self.output/"run_manifest.json").read_text())
            if manifest["identity"] != identity:
                raise ValueError("Resume configuration, code, tokenizer or checkpoint identity changed")
            if not self.path.exists():
                raise ValueError("No committed checkpoint; use a fresh output or explicit --overwrite")
            with self.path.open("rb") as stream:
                saved = pickle.load(stream)
            if saved["identity"] != identity:
                raise ValueError("Checkpoint identity differs from its manifest")
            self.restored = saved["state"]
            for name in STREAMS:
                path = self.output/name
                offset = saved["offsets"][name]
                if not path.exists() or path.stat().st_size < offset:
                    raise ValueError("Committed collection stream is missing or shorter than its checkpoint")
                with path.open("r+b") as stream:
                    stream.truncate(offset)
        for name in STREAMS:
            self.streams[name] = (self.output/name).open("ab")

    def write(self, name, record):
        self.streams[name].write((json.dumps(record, ensure_ascii=False, allow_nan=False, separators=(",", ":"))+"\n").encode())
        self.streams[name].flush()

    def commit(self, state):
        offsets = {}
        for name, stream in self.streams.items():
            stream.flush()
            os.fsync(stream.fileno())
            offsets[name] = stream.tell()
        atomic_bytes(self.path, pickle.dumps(dict(identity=self.identity, offsets=offsets, state=state), protocol=5))

    def close(self):
        for stream in self.streams.values():
            stream.close()


def directory_fingerprint(path):
    """Hash tokenizer files, not the multi-gigabyte model weights."""
    root = Path(path).resolve()
    names = ("tokenizer_config.json", "tokenizer.json", "vocab.json", "merges.txt", "special_tokens_map.json", "chat_template.jinja")
    files = {name: hashlib.sha256((root/name).read_bytes()).hexdigest() for name in names if (root/name).is_file()}
    if "tokenizer_config.json" not in files:
        raise ValueError("A complete local target tokenizer directory is required")
    return dict(path=str(root), files=files)


class TokenizerWorker:
    def __init__(self, tokenizer, *, python=None, max_tokens=4096, continuation_groups=2):
        root = Path(__file__).resolve().parents[1]
        self.process = subprocess.Popen([
            python or sys.executable, "-u", "-m", "LLM.finetune.guidance_view",
            "--tokenizer", str(Path(tokenizer).resolve()), "--max-tokens", str(max_tokens),
            "--continuation-groups", str(continuation_groups)], cwd=root,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, encoding="utf-8")
        try:
            self.ready = self._read()
            if self.ready.get("event") != "tokenizer_ready":
                raise RuntimeError("Tokenizer worker did not report readiness")
        except BaseException:
            self.close()
            raise

    def _read(self):
        if not select.select([self.process.stdout], [], [], 600)[0]:
            raise TimeoutError("Tokenizer worker did not respond within 600 seconds")
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError("Tokenizer worker exited; check its dependency/error output")
        value = json.loads(line)
        if value.get("error"):
            raise RuntimeError("Tokenizer worker: "+value["error"])
        return value

    def views(self, payload, guidance, selected):
        self.process.stdin.write(json.dumps(dict(payload=payload, guidance=guidance, selected=selected), ensure_ascii=False)+"\n")
        self.process.stdin.flush()
        return self._read()

    def close(self):
        if self.process.stdin and not self.process.stdin.closed:
            try:
                self.process.stdin.close()
            except BrokenPipeError:
                pass
        try:
            self.process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        if self.process.stdout:
            self.process.stdout.close()
