"""Run persistence, restart integrity and dashboard tests."""

import dataclasses
import json
import zipfile

import pandas as pd
import pytest

from finance_sim.configuration import RunConfig
from finance_sim.results import paired_amanda_comparison
from finance_sim.scenarios import Scenario
from finance_sim.storage import read_table
from finance_sim.workflow import execute


def test_resume_matches_uninterrupted(tmp_path):
    config = RunConfig(paths=2, batch_size=1, threads=2, generate_reports=False)
    scenarios = [Scenario(births=(2029,)), Scenario(births=(2029,), stop_after_birth=1)]
    partial = tmp_path / "partial"
    full = tmp_path / "full"
    manifest = execute(config, partial, selected=scenarios, maximum_batches=1)
    assert manifest["status"] == "partial" and manifest["completed_scenarios"] == 1
    execute(config, partial, selected=scenarios, resume=True)
    execute(config, full, selected=scenarios)
    pd.testing.assert_frame_equal(read_table(partial, "terminal"), read_table(full, "terminal"))
    assert len(paired_amanda_comparison(read_table(full, "summary"))) == 1


def test_resume_rejects_changed_input(tmp_path):
    config = RunConfig(paths=2, batch_size=1, threads=2, generate_reports=False)
    scenario = [Scenario()]
    execute(config, tmp_path / "run", selected=scenario, maximum_batches=1)
    config.seed += 1
    with pytest.raises(ValueError, match="changed inputs"):
        execute(config, tmp_path / "run", selected=scenario, resume=True)


def test_existing_run_not_overwritten(tmp_path):
    config = RunConfig(paths=2, threads=2, generate_reports=False)
    execute(config, tmp_path / "run", preset="baseline")
    with pytest.raises(FileExistsError):
        execute(config, tmp_path / "run", preset="baseline")


def test_resume_recovers_failed_reports_without_resimulation(tmp_path, monkeypatch):
    config = RunConfig(paths=2, threads=2)
    folder = tmp_path / "run"

    def fail_report(folder):
        raise RuntimeError("report interrupted")

    monkeypatch.setattr("finance_sim.reporting.generate_reports", fail_report)
    with pytest.raises(RuntimeError, match="report interrupted"):
        execute(config, folder, preset="baseline")
    before = json.loads((folder / "manifest.json").read_text())
    assert before["status"] == "complete" and before["reports_status"] == "failed"

    def reject_simulation(*args):
        pytest.fail("Completed simulation was repeated")

    reports = []
    monkeypatch.setattr("finance_sim.workflow.simulate", reject_simulation)
    monkeypatch.setattr("finance_sim.reporting.generate_reports", reports.append)
    after = execute(config, folder, preset="baseline", resume=True)
    assert reports == [folder]
    assert after["reports_status"] == "complete" and "report_error" not in after
    assert after["parts"] == before["parts"]
    assert after["runtime_seconds"] == before["runtime_seconds"]
    execute(config, folder, preset="baseline", resume=True)
    assert reports == [folder]


def test_completed_resume_checks_partitions_and_archives_entrypoints(tmp_path):
    config = RunConfig(paths=2, threads=2, generate_reports=False)
    folder = tmp_path / "run"
    manifest = execute(config, folder, preset="baseline")
    with zipfile.ZipFile(folder / "engine_source.zip") as archive:
        assert {"app.py", "dashboard_insights.py", "salary_support.py"} <= set(archive.namelist())
        assert "config.py" not in archive.namelist()
    partition = folder / "terminal" / manifest["parts"][0]["file"]
    partition.write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="missing or corrupted"):
        execute(config, folder, preset="baseline", resume=True)
