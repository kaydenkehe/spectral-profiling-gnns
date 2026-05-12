#!/usr/bin/env python3
"""
Build pushable release folders for architecture + Jacobi spatial-mask experiments.

Architecture: per-dataset summary.csv (same schema as train_spectral_massive paper runs),
              plus optional summary_all.csv.

Jacobi: per-dataset summary.csv + summary_details.csv; for Computers/CS/RomanEmpire,
        merge May 11 (K=4 only in practice) with May 12 k10_long (K=10).
"""
from __future__ import annotations

import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
RELEASE = REPO / "release"

ARCH_OUT = RELEASE / "architecture_spatial_masks"
JAC_OUT = RELEASE / "jacobi_spatial_masks"

ARCH_PRIMARY = REPO / "spectral_massive_spatial_masks_retry_20260511-232502"
ARCH_QUESTIONS = (
    REPO
    / "spectral_massive_spatial_masks_retry_anygpu_b15_20260512-105259"
    / "Questions"
    / "array_12565533_18"
    / "paper_massive_20260512-123438"
)
ARCH_CS_LEGACY = (
    REPO
    / "spectral_massive_spatial_masks_retry_anygpu_b15_20260512-105259"
    / "CS"
    / "array_12565533_7"
    / "paper_massive_20260512-122445"
)
ARCH_CS_PATCH = REPO / "spectral_massive_spatial_masks_cs_missing_b10_20260512-145509"
ARCH_PHYSICS_PARTIAL = (
    REPO
    / "spectral_massive_spatial_masks_retry_anygpu_b15_20260512-105259"
    / "Physics"
    / "array_12565533_8"
    / "paper_massive_20260512-122642"
)

JAC_MAY11 = REPO / "jacobi_ab_sweep_spatial_masks_retry_20260511-232502"
JAC_K10_LONG = REPO / "jacobi_ab_sweep_spatial_masks_retry_k10_long_20260512-105141"

DATASETS_ARCH = [
    "Cora",
    "PubMed",
    "Photo",
    "Texas",
    "Chameleon",
    "CiteSeer",
    "Computers",
    "CS",
    "Physics",
    "Cornell",
    "Wisconsin",
    "Actor",
    "WikiCS",
    "Squirrel",
    "RomanEmpire",
    "AmazonRatings",
    "Minesweeper",
    "Tolokers",
    "Questions",
]

# Jacobi May-11 run is K=4 for these four; May-12 long run supplies K=10.
JAC_MERGE_K4_K10 = ("Computers", "CS", "RomanEmpire")


def _git_rev() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"],
            text=True,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8", errors="replace") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})


def _concat_csv_files(paths: list[Path], out: Path) -> tuple[list[str], int]:
    all_rows: list[dict[str, str]] = []
    fieldnames: list[str] = []
    for p in paths:
        if not p.exists():
            raise FileNotFoundError(p)
        with p.open(newline="", encoding="utf-8", errors="replace") as f:
            rdr = csv.DictReader(f)
            if not fieldnames:
                fieldnames = list(rdr.fieldnames or [])
            for row in rdr:
                all_rows.append(dict(row))
    if not fieldnames and all_rows:
        fieldnames = list(all_rows[0].keys())
    _write_csv(out, fieldnames, all_rows)
    return fieldnames, len(all_rows)


def _find_one_summary(base: Path, dataset: str) -> Path | None:
    hits = sorted(base.glob(f"{dataset}/array_*/paper_massive_*/summary.csv"))
    if not hits:
        hits = sorted(base.glob(f"{dataset}/array_*/**/summary.csv"))
    return hits[-1] if hits else None


def _find_jacobi_may11(dataset: str) -> tuple[Path | None, Path | None]:
    sums = sorted((JAC_MAY11 / dataset).glob("array_*/**/summary.csv"))
    dets = sorted((JAC_MAY11 / dataset).glob("array_*/**/summary_details.csv"))
    return (sums[-1] if sums else None), (dets[-1] if dets else None)


def _find_jacobi_k10(dataset: str) -> tuple[Path | None, Path | None]:
    sums = sorted((JAC_K10_LONG / dataset).glob("array_*/**/summary.csv"))
    dets = sorted((JAC_K10_LONG / dataset).glob("array_*/**/summary_details.csv"))
    return (sums[-1] if sums else None), (dets[-1] if dets else None)


def _merge_cs_architecture() -> dict:
    legacy = ARCH_CS_LEGACY / "summary.csv"
    bern = ARCH_CS_PATCH / "bern" / "paper_massive_20260512-153542" / "summary.csv"
    jac = ARCH_CS_PATCH / "jacobi" / "paper_massive_20260512-154137" / "summary.csv"
    cheb = ARCH_CS_PATCH / "cheb_k10" / "paper_massive_20260512-153502" / "summary.csv"
    paths = [legacy, bern, jac, cheb]
    missing = [str(p.relative_to(REPO)) for p in paths if not p.exists()]
    if missing:
        print("ERROR: missing CS patch source(s):", missing, file=sys.stderr)
        sys.exit(1)

    parts = [_read_csv(p) for p in paths]
    ids: set[str] = set()
    merged: list[dict[str, str]] = []
    fieldnames: list[str] | None = None
    for label, rows in zip(("legacy", "bern", "jacobi_model", "cheb_k10"), parts):
        if not rows:
            print(f"WARN: empty {label}", file=sys.stderr)
            continue
        if fieldnames is None:
            fieldnames = list(rows[0].keys())
        for r in rows:
            rid = r.get("id", "")
            if rid in ids:
                raise SystemExit(f"duplicate id in CS merge: {rid!r} from {label}")
            ids.add(rid)
            merged.append(r)
    assert fieldnames is not None
    out = ARCH_OUT / "CS" / "summary.csv"
    _write_csv(out, fieldnames, merged)
    # Prefer legacy config as baseline
    cfg_src = ARCH_CS_LEGACY / "config.json"
    if cfg_src.exists():
        shutil.copy2(cfg_src, ARCH_OUT / "CS" / "config.json")
    info = {
        "dataset": "CS",
        "rows": len(merged),
        "expected_rows": 1440,
        "complete": len(merged) >= 1440,
        "sources": [str(p.relative_to(REPO)) for p in paths],
    }
    if len(merged) < 1440:
        info["note"] = (
            "Incomplete until Cheb K=10 patch finishes; re-run this script after job completion."
        )
    return info


def build_architecture() -> dict:
    shutil.rmtree(ARCH_OUT, ignore_errors=True)
    ARCH_OUT.mkdir(parents=True)
    manifest: dict = {"kind": "architecture_spatial_masks", "datasets": {}}
    all_summaries: list[Path] = []

    for ds in DATASETS_ARCH:
        if ds == "CS":
            info = _merge_cs_architecture()
            manifest["datasets"]["CS"] = info
            all_summaries.append(ARCH_OUT / "CS" / "summary.csv")
            continue
        if ds == "Questions":
            src = ARCH_QUESTIONS / "summary.csv"
            dst = ARCH_OUT / "Questions" / "summary.csv"
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            cfg = ARCH_QUESTIONS / "config.json"
            if cfg.exists():
                shutil.copy2(cfg, ARCH_OUT / "Questions" / "config.json")
            n = sum(1 for _ in open(dst)) - 1
            manifest["datasets"]["Questions"] = {
                "rows": n,
                "expected_rows": 1440,
                "summary": str(src.relative_to(REPO)),
            }
            all_summaries.append(dst)
            continue
        if ds == "Physics":
            # Partial only; park under partial/
            src = ARCH_PHYSICS_PARTIAL / "summary.csv"
            dst = ARCH_OUT / "partial" / "Physics" / "summary.csv"
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            cfg = ARCH_PHYSICS_PARTIAL / "config.json"
            if cfg.exists():
                shutil.copy2(cfg, ARCH_OUT / "partial" / "Physics" / "config.json")
            n = sum(1 for _ in open(dst)) - 1
            manifest["datasets"]["Physics"] = {
                "rows": n,
                "note": "incomplete run (OOM); not part of full 1440 grid",
                "summary": str(src.relative_to(REPO)),
            }
            continue

        src = _find_one_summary(ARCH_PRIMARY, ds)
        if src is None:
            raise SystemExit(f"no architecture summary for {ds} under {ARCH_PRIMARY}")
        dst = ARCH_OUT / ds / "summary.csv"
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        cfg = src.parent / "config.json"
        if cfg.exists():
            shutil.copy2(cfg, ARCH_OUT / ds / "config.json")
        n = sum(1 for _ in open(dst)) - 1
        manifest["datasets"][ds] = {
            "rows": n,
            "expected_rows": 1440,
            "summary": str(src.relative_to(REPO)),
        }
        all_summaries.append(dst)

    # One combined file for convenience (same schema).
    combined_rows: list[dict[str, str]] = []
    fields: list[str] | None = None
    for p in sorted(all_summaries, key=lambda x: x.parts[-2]):
        rows = _read_csv(p)
        if not rows:
            continue
        if fields is None:
            fields = list(rows[0].keys())
        combined_rows.extend(rows)
    if fields:
        _write_csv(ARCH_OUT / "summary_all.csv", fields, combined_rows)

    readme = f"""# Architecture sweep (spatial masks)

- One folder per dataset with `summary.csv`, matching `spectral/train_spectral_massive.py` outputs.
- **CS** is merged from multiple Slurm runs (GPRGNN/Cheb K=4, then BernNet, JacobiConv, Cheb K=10 patches under `spectral_massive_spatial_masks_cs_missing_b10_*`).
- **Questions** is taken from the `retry_anygpu_b15` run (complete 1440 rows); the May 11 retry folder only had a partial Questions file.
- **Physics** is under `partial/Physics/` (OOM; incomplete).
- `summary_all.csv` concatenates all per-dataset summaries (excludes `partial/Physics`).

Git revision: `{_git_rev()}
`
"""
    (ARCH_OUT / "README.md").write_text(readme, encoding="utf-8")
    manifest["git_revision"] = _git_rev()
    manifest["summary_all_rows"] = len(combined_rows)
    (ARCH_OUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def build_jacobi() -> dict:
    shutil.rmtree(JAC_OUT, ignore_errors=True)
    JAC_OUT.mkdir(parents=True)
    manifest: dict = {"kind": "jacobi_spatial_masks", "datasets": {}}

    for ds_dir in sorted([p for p in JAC_MAY11.iterdir() if p.is_dir()]):
        ds = ds_dir.name
        sum_m11, det_m11 = _find_jacobi_may11(ds)
        sum_12, det_12 = _find_jacobi_k10(ds)

        if ds in JAC_MERGE_K4_K10:
            if sum_m11 is None or det_m11 is None:
                raise SystemExit(f"Jacobi May 11 missing for {ds}")
            if sum_12 is None or det_12 is None:
                raise SystemExit(f"Jacobi K=10 long missing for {ds}")
            (JAC_OUT / ds).mkdir(parents=True, exist_ok=True)
            _concat_csv_files([sum_m11, sum_12], JAC_OUT / ds / "summary.csv")
            _concat_csv_files([det_m11, det_12], JAC_OUT / ds / "summary_details.csv")
            cfg11 = sum_m11.parent / "summary_config.json"
            if cfg11.exists():
                shutil.copy2(cfg11, JAC_OUT / ds / "summary_config_may11.json")
            cfg12 = sum_12.parent / "summary_config.json"
            if cfg12.exists():
                shutil.copy2(cfg12, JAC_OUT / ds / "summary_config_k10_long.json")
            n_s = sum(1 for _ in (JAC_OUT / ds / "summary.csv").open()) - 1
            manifest["datasets"][ds] = {
                "rows_summary": n_s,
                "expected_summary": 5000,
                "sources": [
                    str(sum_m11.relative_to(REPO)),
                    str(sum_12.relative_to(REPO)),
                ],
                "merge": "May11 (K=4 grid) + May12 k10_long (K=10 grid)",
            }
            continue

        if ds == "Physics":
            if sum_m11 is None or det_m11 is None:
                raise SystemExit("Jacobi Physics May 11 missing")
            (JAC_OUT / ds).mkdir(parents=True, exist_ok=True)
            shutil.copy2(sum_m11, JAC_OUT / ds / "summary.csv")
            shutil.copy2(det_m11, JAC_OUT / ds / "summary_details.csv")
            cfg = sum_m11.parent / "summary_config.json"
            if cfg.exists():
                shutil.copy2(cfg, JAC_OUT / ds / "summary_config.json")
            n_s = sum(1 for _ in (JAC_OUT / ds / "summary.csv").open()) - 1
            manifest["datasets"]["Physics"] = {
                "rows_summary": n_s,
                "note": "K=10 long run canceled; only May 11 (K=4) retained",
                "sources": [str(sum_m11.relative_to(REPO))],
            }
            continue

        # Default: full May 11 run (already contains K=4 and K=10 for most datasets)
        if sum_m11 is None or det_m11 is None:
            raise SystemExit(f"Jacobi May 11 missing for {ds}")
        (JAC_OUT / ds).mkdir(parents=True, exist_ok=True)
        shutil.copy2(sum_m11, JAC_OUT / ds / "summary.csv")
        shutil.copy2(det_m11, JAC_OUT / ds / "summary_details.csv")
        cfg = sum_m11.parent / "summary_config.json"
        if cfg.exists():
            shutil.copy2(cfg, JAC_OUT / ds / "summary_config.json")
        n_s = sum(1 for _ in (JAC_OUT / ds / "summary.csv").open()) - 1
        manifest["datasets"][ds] = {
            "rows_summary": n_s,
            "sources": [str(sum_m11.relative_to(REPO))],
        }

    readme = f"""# Jacobi A/B sweep (spatial masks)

- Per-dataset `summary.csv` and `summary_details.csv`, matching `spectral/jacobi_ab_sweep_massive.py`.
- **Computers, CS, RomanEmpire:** merged from
  - `jacobi_ab_sweep_spatial_masks_retry_20260511-232502` (**K=4** block, 2500 summary / 25000 detail rows)
  - `jacobi_ab_sweep_spatial_masks_retry_k10_long_20260512-105141` (**K=10** block, same sizes)
- **All other datasets:** taken entirely from the May 11–12 spatial-masks retry (both K already in one run, except where noted).
- **Physics:** only the **K=4** May 11 run is included (K=10 rerun was canceled).

Git revision: `{_git_rev()}
`
"""
    (JAC_OUT / "README.md").write_text(readme, encoding="utf-8")
    manifest["git_revision"] = _git_rev()
    (JAC_OUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    RELEASE.mkdir(exist_ok=True)
    am = build_architecture()
    jm = build_jacobi()
    print("Wrote", ARCH_OUT)
    print("Wrote", JAC_OUT)
    print(
        "CS architecture rows:",
        am["datasets"].get("CS", {}).get("rows"),
        "/",
        am["datasets"].get("CS", {}).get("expected_rows"),
    )


if __name__ == "__main__":
    main()
