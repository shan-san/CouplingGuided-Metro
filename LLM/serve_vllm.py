"""Launch the native vLLM server with the trained adapter and its tokenizer."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import os
from pathlib import Path
import shlex
import sys


def build_command(args):
    adapter = Path(args.adapter).expanduser().resolve()
    for name in ("adapter_config.json", "adapter_model.safetensors", "sft_manifest.json", "tokenizer_config.json", "tokenizer.json"):
        if not (adapter / name).is_file():
            raise ValueError(f"Missing adapter asset: {adapter / name}")
    manifest = json.loads((adapter / "sft_manifest.json").read_text())
    config = json.loads((adapter / "adapter_config.json").read_text())
    if manifest.get("input_encoding") != "metro_guidance_group_v2":
        raise ValueError("Expected a native grouped-guidance SFT adapter")
    if manifest.get("limited_run"):
        raise ValueError("A limited smoke-test adapter is not a completed SFT deployment")
    context = manifest.get("max_length")
    if type(context) is not int or not 0 < args.max_model_len <= context:
        raise ValueError("Serving context must not exceed the SFT context limit")
    rank = config.get("r")
    if config.get("peft_type") != "LORA" or type(rank) is not int or rank <= 0:
        raise ValueError("Expected a LoRA adapter with a positive rank")
    if args.model_name == args.base_name:
        raise ValueError("The adapter and base model must have different served names")
    if not 0 < args.port < 65536 or args.max_num_seqs < 1:
        raise ValueError("Invalid port or concurrent sequence limit")
    if not math.isfinite(args.gpu_memory_utilization) or not 0 < args.gpu_memory_utilization < 1:
        raise ValueError("GPU memory utilization must lie between zero and one")
    base = Path(args.base_model or config["base_model_name_or_path"]).expanduser().resolve()
    base_config = json.loads((base / "config.json").read_text())
    quant = base_config.get("quantization_config", {})
    quant_method = quant.get("quant_method")
    if quant_method != "bitsandbytes":
        raise ValueError(f"This deployment expects the existing BitsAndBytes base, not {quant_method!r}")
    if quant_method == "bitsandbytes" and not quant.get("load_in_4bit", False):
        raise ValueError("Expected the training-time 4-bit base checkpoint")
    command = ["vllm", "serve", str(base), "--host", args.host, "--port", str(args.port),
        "--served-model-name", args.base_name, "--tokenizer", str(adapter),
        "--dtype", "bfloat16", "--tensor-parallel-size", "1",
        "--max-model-len", str(args.max_model_len), "--max-num-seqs", str(args.max_num_seqs),
        "--gpu-memory-utilization", str(args.gpu_memory_utilization), "--enable-lora",
        "--max-loras", "1", "--max-lora-rank", str(rank),
        "--lora-modules", f"{args.model_name}={adapter}", "--generation-config", "vllm",
        "--language-model-only", "--seed", "111"]
    if quant_method == "bitsandbytes":
        command.extend(["--quantization", "bitsandbytes", "--load-format", "bitsandbytes"])
    if args.enforce_eager:
        command.append("--enforce-eager")
    return command


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--base-model")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--model-name", default="metro-slm-v9")
    parser.add_argument("--base-name", default="metro-slm-base")
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--max-num-seqs", type=int, default=4)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--deployment-record", help="Write the launch configuration and package versions")
    args = parser.parse_args(argv)
    command = build_command(args)
    if args.dry_run:
        print(shlex.join(command))
        return command
    if os.environ.get("CUDA_VISIBLE_DEVICES", "").strip() in ("", "-1"):
        raise RuntimeError("Set CUDA_VISIBLE_DEVICES to the GPU assigned by Slurm before serving")
    executable = Path(sys.executable).with_name("vllm")
    if not executable.is_file():
        raise RuntimeError("Activate the separate vLLM environment before serving")
    versions = {}
    for name in ("vllm", "vllm-bnb-plugin", "torch", "transformers", "bitsandbytes"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    if "--quantization" in command and versions["vllm-bnb-plugin"] is None:
        raise RuntimeError("Install vllm-bnb-plugin in the serving environment")
    if args.deployment_record:
        record = Path(args.deployment_record)
        record.parent.mkdir(parents=True, exist_ok=True)
        record.write_text(json.dumps(dict(command=command, versions=versions,
            cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES")), indent=2) + "\n")
    print(shlex.join(command), flush=True)
    os.execv(str(executable), command)


if __name__ == "__main__":
    main()
