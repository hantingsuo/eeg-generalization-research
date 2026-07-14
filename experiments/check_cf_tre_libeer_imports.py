"""Import and forward-pass the frozen LibEER deep references without training."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import torch


EXPECTED_COMMIT = "39dc27e504e14138767b87ce8bce485380fd4f5a"
DEFAULT_LIBEER_ROOT = Path(
    os.environ.get("LIBEER_ROOT", "external/LibEER")
)


def check_libeer_models(libeer_root: Path) -> dict[str, Any]:
    root = libeer_root.expanduser().resolve()
    if not (root / "models").is_dir() or not (root / "config/model_param").is_dir():
        raise FileNotFoundError(f"invalid LibEER root: {root}")
    commit = subprocess.run(
        ["git", "-C", str(root.parent), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip().lower()
    if commit != EXPECTED_COMMIT:
        raise RuntimeError(f"LibEER commit mismatch: {commit}")

    previous_cwd = Path.cwd()
    sys.path.insert(0, str(root))
    try:
        os.chdir(root)
        definitions = (
            ("DGCNN", root / "models/DGCNN.py", "DGCNN"),
            ("GCBNet", root / "models/GCBNet.py", "GCBNet"),
            ("GCBNet_BLS", root / "models/GCBNet_BLS.py", "GCBNet_BLS"),
        )
        input_tensor = torch.zeros((2, 62, 5), dtype=torch.float32)
        models: list[dict[str, Any]] = []
        for name, module_file, class_name in definitions:
            module_name = f"_cf_tre_libeer_{name.lower()}"
            spec = importlib.util.spec_from_file_location(module_name, module_file)
            if spec is None or spec.loader is None:
                raise ImportError(f"cannot load {module_file}")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            model_class = getattr(module, class_name)
            model = model_class(num_electrodes=62, in_channels=5, num_classes=3)
            model.eval()
            with torch.no_grad():
                output = model(input_tensor)
            if tuple(output.shape) != (2, 3) or not bool(torch.isfinite(output).all()):
                raise RuntimeError(f"{name} forward-pass failure: {tuple(output.shape)}")
            models.append(
                {
                    "name": name,
                    "module_file": str(module_file),
                    "class": class_name,
                    "output_shape": list(output.shape),
                    "finite_output": True,
                    "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
                }
            )
    finally:
        os.chdir(previous_cwd)
        if sys.path and sys.path[0] == str(root):
            sys.path.pop(0)

    return {
        "protocol": "cf_tre_seed_family_v1",
        "status": "PASS",
        "role": "import_and_forward_only_no_training",
        "libeer_root": str(root),
        "libeer_commit": commit,
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "models": models,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--libeer-root", type=Path, default=DEFAULT_LIBEER_ROOT)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/cf_tre/g0a/libeer_deep_import_forward_v1.json"),
    )
    args = parser.parse_args()
    report = check_libeer_models(args.libeer_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite audit artifact: {args.output}")
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
