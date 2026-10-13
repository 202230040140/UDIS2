"""Stitch one image pair with the pretrained UDIS++ two-stage pipeline."""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from pathlib import Path

import cv2

from reproduce.paths import COMPOSE_CODES, DEFAULT_PYTHON, WARP_CODES


def ensure_trailing_sep(path: Path) -> str:
    text = str(path.resolve())
    if not text.endswith(("\\", "/")):
        text += os.sep
    return text


def is_resource_error(exc: BaseException) -> bool:
    message = str(exc).lower()
    return any(text in message for text in ('outofmemoryerror', 'out of memory',
               'not enough memory', 'defaultcpuallocator', 'unable to allocate'))


def stage_inputs(left_path: str, right_path: str, work_dir: Path, max_side: int) -> None:
    work_dir.mkdir(parents=True, exist_ok=True)
    for src, name in ((left_path, "input1.jpg"), (right_path, "input2.jpg")):
        image = cv2.imread(src)
        if image is None:
            raise RuntimeError("failed to read image: " + src)
        height, width = image.shape[:2]
        if max_side > 0 and max(height, width) > max_side:
            scale = max_side / float(max(height, width))
            image = cv2.resize(
                image,
                (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
                interpolation=cv2.INTER_AREA,
            )
        if not cv2.imwrite(str(work_dir / name), image):
            raise RuntimeError("failed to write staged image: " + name)


def run_stage(python: str, cwd: Path, script: str, args: list[str]) -> None:
    proc = subprocess.run(
        [python, script, *args],
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if proc.returncode != 0:
        tail = proc.stdout[-6000:] if proc.stdout else ""
        raise RuntimeError(script + " failed (exit " + str(proc.returncode) + "):\n" + tail)


def _run_once(left_path, right_path, work, out, gpu, max_side, max_iter, python, path_arg):
    run_stage(
        python,
        WARP_CODES,
        "test_other.py",
        [
            "--gpu",
            gpu,
            "--max_iter",
            str(max_iter),
            "--path",
            path_arg,
            "--img1_name",
            "input1.jpg",
            "--img2_name",
            "input2.jpg",
        ],
    )
    run_stage(python, COMPOSE_CODES, "test_other.py", ["--gpu", gpu, "--path", path_arg])
    composition = work / "composition.jpg"
    if not composition.exists():
        raise RuntimeError("composition output missing: " + str(composition))
    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(composition, out)


def stitch_pair(
    left_path: str,
    right_path: str,
    out_path: str | Path,
    work_dir: str | Path,
    *,
    gpu: str = "0",
    max_side: int = 768,
    max_iter: int = 50,
    python: str = DEFAULT_PYTHON,
    keep_work: bool = False,
) -> Path:
    # The author's adaptation script silently trains from scratch when warp
    # weights are absent. That is not pretrained UDIS++ inference.
    for model_dir in (WARP_CODES.parent / "model", COMPOSE_CODES.parent / "model"):
        if not any(model_dir.glob("*.pth")):
            raise FileNotFoundError("Required pretrained checkpoint missing: " + str(model_dir))
    work = Path(work_dir)
    out = Path(out_path)
    sides = []
    side = max_side if max_side > 0 else 768
    while side >= 384:
        if side not in sides:
            sides.append(side)
        side //= 2
    last_error = None
    for attempt_side in sides:
        if work.exists():
            shutil.rmtree(work, ignore_errors=True)
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        stage_inputs(left_path, right_path, work, max_side=attempt_side)
        path_arg = ensure_trailing_sep(work)
        try:
            _run_once(left_path, right_path, work, out, gpu, attempt_side, max_iter, python, path_arg)
            if not keep_work:
                shutil.rmtree(work, ignore_errors=True)
            return out
        except RuntimeError as exc:
            last_error = exc
            if not is_resource_error(exc):
                raise
    if last_error is not None:
        raise last_error
    raise RuntimeError("stitch_pair failed without error")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left", required=True)
    parser.add_argument("--right", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--max-side", type=int, default=768)
    parser.add_argument("--max-iter", type=int, default=50)
    parser.add_argument("--python", default=DEFAULT_PYTHON)
    args = parser.parse_args()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.set_device(int(args.gpu))
    except Exception:
        pass
    out = stitch_pair(
        args.left,
        args.right,
        args.out,
        args.work_dir,
        gpu=args.gpu,
        max_side=args.max_side,
        max_iter=args.max_iter,
        python=args.python,
    )
    print("Wrote " + str(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


