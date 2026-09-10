"""Run exhaustive or explicitly named finance scenario presets."""

import argparse
import dataclasses

from pathlib import Path

from finance_sim.configuration import load_config
from finance_sim.scenarios import count_scenarios
from finance_sim.workflow import execute
from showcase import update_showcase


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.py")
    parser.add_argument("--output", default="outputs/run")
    parser.add_argument("--preset", choices=("standard", "full", "baseline", "demo", "family"), default="standard")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--paths", type=int)
    parser.add_argument("--no-reports", action="store_true")
    parser.add_argument("--no-showcase", action="store_true", help="Do not refresh the versioned README figure gallery")
    parser.add_argument("--maximum-batches", type=int, help="Explicitly pause after this many batches; output remains partial")
    parser.add_argument("--career", action="append", help="Explicitly restrict full-grid career axis")
    parser.add_argument("--location", action="append", help="Explicitly restrict full-grid location axis")
    args = parser.parse_args()
    if (args.career or args.location) and args.preset != "full":
        parser.error("--career and --location select full-grid axes; use --preset full")
    config = load_config(args.config)
    if args.paths is not None:
        config.paths = args.paths
    if args.no_reports:
        config.generate_reports = False
    for name in ("career", "location"):
        if getattr(args, name):
            setattr(config.grid, name + "s", tuple(getattr(args, name)))
    config.validate()
    count = count_scenarios(config, args.preset)
    print(f"Preset: {args.preset}. Exhaustive configured coverage: {count:,} scenarios, {count*config.paths:,} paths.")
    print(f"Monthly household transitions including bridge: {count*config.paths*((config.end_year-2026)*12+3):,}.")
    print("Runtime depends on this grid. No automatic reduction or sampling of scenario combinations.")
    if args.dry_run:
        print(f"Approximate uncompressed terminal samples: {count*config.paths*70*8/1e9:,.2f} GB; monthly bands add storage.")
        return
    manifest = execute(config, Path(args.output), args.preset, args.resume, maximum_batches=args.maximum_batches)
    if not args.no_showcase and manifest["status"] == "complete" and manifest["reports_status"] == "complete":
        folder = update_showcase()
        if folder:
            print(f"README gallery refreshed from {folder.name}. Commit README.md and docs/results/latest/ to update GitHub.")


if __name__ == "__main__":
    main()
