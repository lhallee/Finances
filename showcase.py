"""Refresh the GitHub README and figure gallery from the latest completed report."""

from __future__ import annotations

import argparse
import json
import shutil

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote


START = "<!-- latest-results:start -->"
END = "<!-- latest-results:end -->"
FEATURED = ("01_", "03_", "10_", "19_", "25_", "26_", "27_")


@dataclass(frozen=True)
class Figure:
    filename: str
    description: str

    @property
    def title(self) -> str:
        return Path(self.filename).stem.split("_", 1)[-1].replace("_", " ").capitalize()


@dataclass(frozen=True)
class CompletedReport:
    folder: Path
    completed_at: datetime
    scenarios: int
    paths_per_scenario: int
    preset: str
    simulation_seconds: float
    report_seconds: float
    figures: tuple[Figure, ...]


def latest_report(outputs: Path) -> CompletedReport | None:
    """Skip partial runs and unfinished/missing figures, including interrupted exports."""
    reports = []
    for manifest_path in outputs.rglob("manifest.json"):
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue  # Another process can still be creating this run.
        if manifest.get("status") != "complete" or manifest.get("reports_status") != "complete":
            continue
        if manifest.get("completed_scenarios", 0) != manifest.get("expected_scenarios"):
            continue
        folder = manifest_path.parent
        if not (folder / "figure_index.json").is_file() or not (folder / "config.json").is_file():
            continue
        index = json.loads((folder / "figure_index.json").read_text(encoding="utf-8"))
        figures = tuple(Figure(row["file"], row["description"]) for row in index)
        if not figures:
            continue
        for figure in figures:
            if Path(figure.filename).name != figure.filename or Path(figure.filename).suffix != ".png":
                raise ValueError(f"Invalid figure filename in {folder}: {figure.filename}")
        if not all((folder / "figures" / figure.filename).is_file() for figure in figures):
            continue
        config = json.loads((folder / "config.json").read_text(encoding="utf-8"))
        # The workflow's final manifest write follows successful report export.
        completed_at = datetime.fromtimestamp(manifest_path.stat().st_mtime, timezone.utc)
        reports.append(CompletedReport(
            folder, completed_at, int(manifest["completed_scenarios"]), int(config["paths"]),
            'explicit selection' if manifest['selection'].get('explicit') is not None else str(manifest["selection"]["preset"]), float(manifest["runtime_seconds"]),
            float(manifest.get("report_seconds", 0)), figures,
        ))
    return max(reports, key=lambda run: (run.completed_at, run.folder.as_posix())) if reports else None


def update_showcase(repo: Path | None = None) -> Path | None:
    """Copy only figures and a small allowlisted summary; never export private ledgers."""
    root = (repo or Path(__file__).resolve().parent).resolve()
    report = latest_report(root / "outputs")
    if report is None:
        return None
    readme_path = root / "README.md"
    readme = readme_path.read_text(encoding="utf-8")
    if (START in readme) != (END in readme) or readme.count(START) > 1 or readme.count(END) > 1:
        raise ValueError("README must contain either one matching results marker pair or neither")
    if START in readme and readme.index(END) < readme.index(START):
        raise ValueError("README results markers are out of order")

    destination = root / "docs" / "results" / "latest"
    images = destination / "images"
    images.mkdir(parents=True, exist_ok=True)
    for figure in report.figures:
        shutil.copyfile(report.folder / "figures" / figure.filename, images / figure.filename)

    relative_run = report.folder.relative_to(root).as_posix()
    metadata = {
        "run": relative_run, "report_completed_utc": report.completed_at.isoformat(),
        "preset": report.preset, "scenarios": report.scenarios,
        "paths_per_scenario": report.paths_per_scenario,
        "total_paths": report.scenarios * report.paths_per_scenario,
        "simulation_and_storage_seconds": report.simulation_seconds,
        "report_seconds": report.report_seconds, "figure_count": len(report.figures),
    }
    (destination / "summary.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    scope = (
        "Conditional simulation results with provisional assumptions, not a probability-weighted forecast. "
        "Trajectory bands are marginal percentiles; company charts use one actual path. "
        "Figures can cover different scenario subsets, described in their captions."
    )
    gallery = ["# Latest simulation figures", "", f"Source run: `{relative_run}`.", "", scope,
               "", "[Model definitions and limitations](../../../MODEL.md)", "",
               "Only report figures and aggregate run metadata are included here. "
               "Workbooks, input files and path ledgers remain local.", ""]
    for figure in report.figures:
        gallery.extend([f"## {figure.title}", "", f"![{figure.title}](images/{quote(figure.filename)})",
                        "", figure.description, ""])
    (destination / "README.md").write_text("\n".join(gallery), encoding="utf-8")

    selection_label = report.preset if report.preset == 'explicit selection' else f'{report.preset} preset'
    lines = [START, "## Latest results", "",
             f"Latest completed report: **`{relative_run}`** ({selection_label}), "
             f"updated **{report.completed_at:%Y-%m-%d %H:%M UTC}**.", "",
             "| Scenarios | Paths per scenario | Total paths | Simulation + storage | Reports | Figures |",
             "| ---: | ---: | ---: | ---: | ---: | ---: |",
             f"| {report.scenarios:,} | {report.paths_per_scenario:,} | {metadata['total_paths']:,} "
             f"| {report.simulation_seconds:.1f} s | {report.report_seconds:.1f} s | {len(report.figures)} |",
             "", scope, "",
             "[View every figure](docs/results/latest/README.md) · "
             "[Run metadata](docs/results/latest/summary.json) · [Model limitations](MODEL.md)", ""]
    featured = [figure for prefix in FEATURED for figure in report.figures if figure.filename.startswith(prefix)]
    for first in range(0, len(featured), 2):
        pair = featured[first:first+2]
        lines.extend(["| " + " | ".join(f.title for f in pair) + " |",
                      "| " + " | ".join("---" for _ in pair) + " |",
                      "| " + " | ".join(f"![{f.title}](docs/results/latest/images/{quote(f.filename)})" for f in pair) + " |", ""])
    lines.extend(["This section and the versioned gallery refresh after successful CLI report runs. "
                  "Commit and push `README.md` and `docs/results/latest/` to display the new results on GitHub. "
                  "Run `python showcase.py` to refresh from existing outputs. "
                  "Partial runs and runs without completed reports do not replace the gallery.", "", END])
    block = "\n".join(lines)
    if START in readme:
        before, remainder = readme.split(START, 1)
        _, after = remainder.split(END, 1)
        updated = before + block + after
    else:
        insertion = readme.find("\n## ")
        if insertion < 0:
            insertion = len(readme)
        updated = readme[:insertion].rstrip() + "\n\n" + block + "\n" + readme[insertion:]
    temporary = readme_path.with_suffix(".md.tmp")
    temporary.write_text(updated, encoding="utf-8")
    temporary.replace(readme_path)
    return report.folder


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    folder = update_showcase()
    print(f"README gallery updated from {folder}" if folder else "No completed reports found; README unchanged.")


if __name__ == "__main__":
    main()
