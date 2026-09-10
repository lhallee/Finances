"""Generate an explicitly subjective probability forecast in a separate folder."""

import argparse

from pathlib import Path

from finance_sim.configuration import config_dict, load_config
from finance_sim.forecast import run_forecast
from finance_sim.storage import write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.py")
    parser.add_argument("--output", default="outputs/forecast")
    parser.add_argument("--paths", type=int)
    args = parser.parse_args()
    config = load_config(args.config)
    if args.paths:
        config.forecast_paths = args.paths
    folder = Path(args.output)
    folder.mkdir(parents=True, exist_ok=False)
    write_json(folder / "config.json", config_dict(config))
    result = run_forecast(config, folder)
    print(f"Saved {len(result):,} probability-weighted outcomes to {folder}")


if __name__ == "__main__":
    main()
