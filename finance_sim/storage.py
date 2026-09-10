"""Atomic run manifests and committed Parquet partitions."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import zipfile

import pandas as pd

from datetime import datetime, timezone
from pathlib import Path
from importlib.metadata import version

from .configuration import RunConfig, config_dict
from .economics import history_fingerprint
from .reference import LIMITATIONS, LOCATIONS, SOURCES


SCHEMA_VERSION = 1


def write_json(path: Path, value: dict | list) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=str, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def code_fingerprint() -> str:
    digest = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob("*.py")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def input_fingerprint(config: RunConfig, selection: dict) -> str:
    payload = {"config": config_dict(config), "selection": selection, "history": history_fingerprint(config)}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def prepare_run(folder: Path, config: RunConfig, selection: dict, count: int, resume: bool) -> dict:
    fingerprint = input_fingerprint(config, selection)
    code = code_fingerprint()
    if (folder / "manifest.json").exists():
        manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
        if not resume:
            raise FileExistsError("Output already contains a run. Use --resume or a new folder.")
        if manifest["input_hash"] != fingerprint or manifest["code_hash"] != code:
            raise ValueError("Cannot resume with changed inputs, history, selection, or engine code")
        for part in manifest["parts"]:
            for kind in ("summary", "terminal", "monthly"):
                path = folder / kind / part["file"]
                if not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest() != part["hashes"][kind]:
                    raise ValueError(f"Committed partition missing or corrupted: {path}")
        return manifest
    if folder.exists() and any(folder.iterdir()):
        raise FileExistsError("Choose an empty output directory")
    for name in ("summary", "terminal", "monthly", "ledgers", "figures"):
        (folder / name).mkdir(parents=True, exist_ok=True)
    resolved = config_dict(config)
    sources = list(SOURCES)
    if config.economics.history_csv:
        source = Path(config.economics.history_csv).resolve()
        shutil.copy2(source, folder / "history.csv")
        resolved["economics"]["history_csv"] = "history.csv"
        metadata = source.parent / "sources.json"
        if metadata.exists():
            shutil.copy2(metadata, folder / "history_metadata.json")
        sources.append({"id": "historical_calibration", "url": "https://fred.stlouisfed.org/", "status": "downloaded_and_aligned",
                        "as_of": config.economics.reference_date, "use": "Run-local history.csv; hashes and missing-data treatment saved with calibration"})
    write_json(folder / "config.json", resolved)
    write_json(folder / "sources.json", sources)
    root = Path(__file__).resolve().parent.parent
    with zipfile.ZipFile(folder / "engine_source.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for source in sorted((root / "finance_sim").glob("*.py")):
            archive.write(source, source.relative_to(root))
        for name in ("simulate.py", "salary_support.py", "forecast.py", "app.py", "showcase.py", "dashboard_charts.py", "dashboard_insights.py", "display_names.py", "scenario_details.py", "requirements.txt"):
            archive.write(root / name, name)
    write_json(folder / "locations.json", {key: vars(value) for key, value in LOCATIONS.items()})
    manifest = {"retirement_budget_model": "lifecycle_v3", "dollar_basis": "2026 USD", "monthly_units": "pathwise deflated before quantiles", "ledger_units": "nominal accounting USD", "schema_version": SCHEMA_VERSION, "status": "running", "created_utc": datetime.now(timezone.utc).isoformat(),
                "expected_scenarios": count, "completed_scenarios": 0, "parts": [], "selection": selection,
                "input_hash": fingerprint, "code_hash": code, "history_hash": history_fingerprint(config),
                "python": platform.python_version(), "limitations": LIMITATIONS, "runtime_seconds": 0.,
                "dependencies": {name: version(name) for name in ("numpy", "pandas", "numba", "pyarrow", "streamlit", "matplotlib")},
                "reports_status": "pending", "scenario_weighting": "conditional; equal scenario mixtures are not forecasts"}
    write_json(folder / "manifest.json", manifest)
    return manifest


def commit_batch(folder: Path, manifest: dict, summaries: list[pd.DataFrame], terminals: list[pd.DataFrame], monthly: list[pd.DataFrame]) -> None:
    name = f"part-{len(manifest['parts']):07d}.parquet"
    hashes = {}
    for kind, frames in (("summary", summaries), ("terminal", terminals), ("monthly", monthly)):
        path = folder / kind / name
        temporary = path.with_suffix(".tmp")
        pd.concat(frames, ignore_index=True).to_parquet(temporary, index=False, compression="zstd")
        os.replace(temporary, path)
        hashes[kind] = hashlib.sha256(path.read_bytes()).hexdigest()
    count = sum(len(frame) for frame in summaries)
    manifest["parts"].append({"file": name, "scenarios": count, "hashes": hashes})
    manifest["completed_scenarios"] += count
    write_json(folder / "manifest.json", manifest)


def read_table(folder: Path, kind: str, scenario_ids: list[str] | None = None, columns: list[str] | None = None, limit: int | None = None) -> pd.DataFrame:
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    if manifest["schema_version"] != SCHEMA_VERSION:
        raise ValueError("Unsupported run schema")
    paths = [folder / kind / part["file"] for part in manifest["parts"]]
    if not paths:
        return pd.DataFrame()
    import pyarrow.dataset as ds
    dataset = ds.dataset(paths, format="parquet")
    expression = ds.field("scenario_id").isin(scenario_ids) if scenario_ids is not None else None
    scanner = dataset.scanner(columns=columns, filter=expression)
    return (scanner.head(limit) if limit is not None else scanner.to_table()).to_pandas()
