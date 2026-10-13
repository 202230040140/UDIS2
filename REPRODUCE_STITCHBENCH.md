# UDIS++ StitchBench General + HD3D Reproduction

Outputs are centralized under `D:\StitchBench_Result\udis2` (StitchBench) and
`D:\HD3D_Dataset_Result\<scene>\pair_XX\udis2\` (HD3D), matching the UDIS layout.

## Environment

- Python: `C:\Users\22499\.venvs\obj-gsp-sam\Scripts\python.exe` (PyTorch + CUDA)
- GPU: `--gpu 0`
- Weights: `Warp/model/*.pth`, `Composition/model/*.pth`
- Default `max_side=768` (required on 11 GB GPUs; OOM pairs retry at 384)

## Smoke test

```powershell
cd C:\Users\22499\Documents\GitHub\UDIS2
powershell -File tools\run_stitchbench_udis2.ps1 -Scene SVA-01_chess -Scene APAP-01_garden -Force
powershell -File tools\run_hd3d_udis2.ps1 -Limit 2
```

## Full runs

```powershell
powershell -File tools\run_stitchbench_udis2.ps1 -SkipExisting
powershell -File tools\run_hd3d_udis2.ps1 -SkipExisting
```

Logs: `D:\StitchBench_Result\udis2\full_run.log`, `D:\HD3D_Dataset_Result\_work\udis2_full_run.log`

## Single pair (debug)

```powershell
python reproduce\udis2_stitch_pair.py `
  --left D:\StitchBench\General\SVA-01_chess\01.JPG `
  --right D:\StitchBench\General\SVA-01_chess\02.JPG `
  --out D:\StitchBench_Result\udis2\_debug\panorama.png `
  --work-dir D:\StitchBench_Result\udis2\_debug\work `
  --gpu 0 --max-side 768
```

Some wide-parallax scenes may fail (mesh blowout / OOM); they are recorded as
`failed` in `metrics.csv` and the batch continues.
