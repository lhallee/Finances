"""GitHub assets must follow the latest successful report without copying inputs."""

import json
import os

from showcase import update_showcase


def make_report(root, name, timestamp, status="complete", reports_status="complete"):
    folder = root / "outputs" / name
    (folder / "figures").mkdir(parents=True)
    (folder / "figures" / "01_net_worth.png").write_bytes(b"test figure")
    (folder / "config.json").write_text(json.dumps({"paths": 256, "private_balance": 123456}))
    (folder / "figure_index.json").write_text(json.dumps([{"file": "01_net_worth.png", "description": "Conditional test outcomes"}]))
    (folder / "manifest.json").write_text(json.dumps({"status": status, "reports_status": reports_status,
        "completed_scenarios": 20, "expected_scenarios": 20, "runtime_seconds": 3., "report_seconds": 1., "selection": {"preset": "family"}}))
    os.utime(folder / "manifest.json", (timestamp, timestamp))
    return folder


def test_latest_complete_gallery_preserves_prose_and_private_inputs(tmp_path):
    (tmp_path / "README.md").write_text("# Example\n\nIntro\n\n## Run locally\n\nInstructions\n")
    make_report(tmp_path, "older", 1000)
    newest = make_report(tmp_path, "newest", 2000)
    make_report(tmp_path, "partial", 3000, status="partial")
    make_report(tmp_path, "unfinished-report", 4000, reports_status="pending")
    assert update_showcase(tmp_path) == newest
    readme = (tmp_path / "README.md").read_text()
    assert "outputs/newest" in readme and "Intro" in readme and "Instructions" in readme
    assert "docs/results/latest/images/01_net_worth.png" in readme
    assert update_showcase(tmp_path) == newest
    assert (tmp_path / "README.md").read_text() == readme
    gallery = tmp_path / "docs/results/latest"
    assert not (gallery / "config.json").exists()
    assert "private_balance" not in (gallery / "summary.json").read_text()


def test_no_reports_leave_readme_unchanged(tmp_path):
    (tmp_path / "README.md").write_text("# Original\n")
    assert update_showcase(tmp_path) is None
    assert (tmp_path / "README.md").read_text() == "# Original\n"


def test_incomplete_figures_do_not_replace_latest(tmp_path):
    (tmp_path / "README.md").write_text("# Example\n")
    older = make_report(tmp_path, "older", 1000)
    broken = make_report(tmp_path, "broken", 2000)
    (broken / "figures/01_net_worth.png").unlink()
    assert update_showcase(tmp_path) == older
