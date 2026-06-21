"""Run UDIS++ on StitchBench General pairs into D:\\StitchBench_Result\\udis2."""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from reproduce.paths import DEFAULT_MANIFEST, DEFAULT_PYTHON, DEFAULT_STITCHBENCH_OUT
from reproduce.udis2_stitch_pair import stitch_pair


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def parse_images(row: dict[str, str]) -> tuple[str, str]:
    files = row["image_files"].split("|")
    if len(files) < 2:
        raise RuntimeError("expected >=2 images for " + row["dataset"])
    data_dir = Path(row["data_dir"])
    return str(data_dir / files[0]), str(data_dir / files[1])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--out-root", default=DEFAULT_STITCHBENCH_OUT)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--python", default=DEFAULT_PYTHON)
    parser.add_argument("--max-side", type=int, default=768)
    parser.add_argument("--max-iter", type=int, default=50)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--scene", action="append")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    rows = read_manifest(Path(args.manifest))
    if args.scene:
        wanted = set(args.scene)
        rows = [row for row in rows if row["dataset"] in wanted]
    if args.limit > 0:
        rows = rows[: args.limit]

    out_root = Path(args.out_root)
    work_root = out_root / "_work"
    work_root.mkdir(parents=True, exist_ok=True)

    metrics_rows = []
    failures = []
    for index, row in enumerate(rows, start=1):
        scene = row["dataset"]
        scene_dir = out_root / scene
        pano_path = scene_dir / "panorama.png"
        print("[{}/{}] {}".format(index, len(rows), scene))

        if args.skip_existing and not args.force and pano_path.exists():
            print("  -> skip (existing panorama)")
            scene_metrics = scene_dir / "metrics.json"
            if scene_metrics.exists():
                prior = json.loads(scene_metrics.read_text(encoding="utf-8"))
                prior["index"] = index
                metrics_rows.append(prior)
            else:
                metrics_rows.append(
                    {
                        "index": index,
                        "scene": scene,
                        "img1": row["image_files"].split("|")[0],
                        "img2": row["image_files"].split("|")[1],
                        "status": "ok",
                        "runtime_seconds": "",
                        "panorama": "panorama.png",
                    }
                )
            continue

        started = time.perf_counter()
        try:
            left, right = parse_images(row)
            stitch_pair(
                left,
                right,
                pano_path,
                work_root / scene,
                gpu=args.gpu,
                max_side=args.max_side,
                max_iter=args.max_iter,
                python=args.python,
            )
            runtime = time.perf_counter() - started
            metrics = {
                "index": index,
                "scene": scene,
                "img1": row["image_files"].split("|")[0],
                "img2": row["image_files"].split("|")[1],
                "status": "ok",
                "runtime_seconds": "{:.2f}".format(runtime),
                "panorama": "panorama.png",
            }
            (scene_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
            metrics_rows.append(metrics)
            print("  -> ok ({:.1f}s)".format(runtime))
        except Exception as exc:
            runtime = time.perf_counter() - started
            failures.append({"scene": scene, "error": str(exc)})
            metrics_rows.append(
                {
                    "index": index,
                    "scene": scene,
                    "img1": row["image_files"].split("|")[0],
                    "img2": row["image_files"].split("|")[1],
                    "status": "failed",
                    "runtime_seconds": "{:.2f}".format(runtime),
                    "panorama": "",
                    "error": str(exc),
                }
            )
            print("  -> failed: {}".format(exc))
            traceback.print_exc()
        finally:
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

    csv_path = out_root / "metrics.csv"
    fields = ["index", "scene", "img1", "img2", "status", "runtime_seconds", "panorama"]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(metrics_rows)

    ok_count = sum(1 for row in metrics_rows if row["status"] in ("ok", "skipped"))
    summary = {
        "method": "udis2",
        "out_root": str(out_root),
        "total": len(rows),
        "ok": ok_count,
        "failed": sum(1 for row in metrics_rows if row["status"] == "failed"),
        "max_side": args.max_side,
        "max_iter": args.max_iter,
        "gpu": args.gpu,
    }
    (out_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\nDone. ok={}/{} metrics={}".format(ok_count, len(rows), csv_path))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())

