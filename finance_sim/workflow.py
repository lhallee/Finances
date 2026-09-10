"""Resumable execution. No scenario is dropped to satisfy a runtime target."""

from __future__ import annotations

import dataclasses
import itertools
import time

import numba
import numpy as np

from pathlib import Path
from collections.abc import Callable, Iterable

from .configuration import RunConfig
from .economics import economic_paths
from .engine import METRICS, simulate
from .results import SimulationResult, summarize
from .scenarios import Scenario, count_scenarios, generate_scenarios
from .storage import commit_batch, prepare_run, write_json


def _finish_reports(config: RunConfig, folder: Path, manifest: dict) -> dict:
    """Retry reports without repeating committed simulations after a failure."""
    if not config.generate_reports or manifest.get("reports_status") == "complete":
        return manifest
    from .reporting import generate_reports
    report_start = time.perf_counter()
    manifest["reports_status"] = "running"
    write_json(folder / "manifest.json", manifest)
    try:
        generate_reports(folder)
        manifest["reports_status"] = "complete"
        manifest.pop("report_error", None)
    except BaseException as error:
        manifest["reports_status"] = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
        manifest["report_error"] = str(error)
        write_json(folder / "manifest.json", manifest)
        raise
    manifest["report_seconds"] = time.perf_counter() - report_start
    write_json(folder / "manifest.json", manifest)
    return manifest


def execute(config: RunConfig, folder: Path, preset: str = "standard", resume: bool = False,
            selected: list[Scenario] | None = None, progress: Callable[[str], None] = print,
            maximum_batches: int | None = None) -> dict:
    config.validate()
    numba.set_num_threads(min(config.threads, numba.config.NUMBA_NUM_THREADS))
    selection = {"preset": preset, "explicit": [s.record() for s in selected] if selected is not None else None}
    count = len(selected) if selected is not None else count_scenarios(config, preset)
    manifest = prepare_run(folder, config, selection, count, resume)
    if manifest["status"] == "complete":
        progress("Simulation is complete; checking report completion.")
        return _finish_reports(config, folder, manifest)
    scenarios = iter(selected) if selected is not None else generate_scenarios(config, preset)
    scenarios = itertools.islice(scenarios, manifest["completed_scenarios"], None)
    start = time.perf_counter()
    completed_at_start = manifest["completed_scenarios"]
    prior_runtime = manifest["runtime_seconds"]
    draw_cache = {}
    progress(f"{count:,} scenarios x {config.paths:,} stochastic paths; {manifest['completed_scenarios']:,} committed.")
    batches = 0
    try:
        while True:
            batch = list(itertools.islice(scenarios, config.batch_size))
            if not batch:
                break
            summaries, terminals, monthly = [], [], []
            for scenario in batch:
                if scenario.macro not in draw_cache:
                    draw_cache[scenario.macro] = economic_paths(config, scenario.macro)  # (p,t,8)
                household, company = simulate(config, scenario, draw_cache[scenario.macro])
                result = SimulationResult(household, company)
                summary, terminal, bands = summarize(result, config, scenario)
                summaries.append(summary)
                terminals.append(terminal)
                monthly.append(bands)
                if manifest["completed_scenarios"] == 0 and len(summaries) <= 3 or scenario.stop_after_birth > 0 and preset == "family":
                    # Real path indices nearest terminal quantiles, never stitched marginal medians.
                    ordered = np.argsort(result.series("net_worth")[:, -1])  # (p,)
                    chosen = np.unique(ordered[np.linspace(0, config.paths-1, min(config.ledger_paths, config.paths), dtype=int)])  # (l,)
                    for index in chosen:
                        result.ledger(int(index), config).to_parquet(folder / "ledgers" / f"{scenario.id}-{index}.parquet", index=False)
            elapsed = time.perf_counter() - start
            manifest["runtime_seconds"] = prior_runtime + elapsed
            commit_batch(folder, manifest, summaries, terminals, monthly)
            done = manifest["completed_scenarios"]
            rate = (done - completed_at_start) / max(elapsed, 1e-9)
            remaining_seconds = (count - done) / max(rate, 1e-9)
            progress(f"{done:,}/{count:,} scenarios saved; {rate:.2f}/s; estimated remaining {remaining_seconds/60:,.1f} min")
            batches += 1
            if maximum_batches is not None and batches >= maximum_batches:
                manifest["status"] = "partial"
                write_json(folder / "manifest.json", manifest)
                return manifest
    except BaseException as error:
        manifest["status"] = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
        manifest["error"] = str(error)
        write_json(folder / "manifest.json", manifest)
        raise
    manifest["status"] = "complete"
    manifest.pop("error", None)
    write_json(folder / "manifest.json", manifest)
    return _finish_reports(config, folder, manifest)
