"""Compute the salary needed when Synthyra never pays the household anything."""

import argparse
from pathlib import Path

from finance_sim.configuration import load_config
from finance_sim.salary_support import run_salary_support, support_scenarios


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.py")
    parser.add_argument("--output", default="outputs/salary-support")
    parser.add_argument("--paths", type=int)
    parser.add_argument("--threads", type=int)
    parser.add_argument("--company-support", choices=("none", "darpa_only"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.paths:
        config.paths = args.paths
    if args.threads:
        config.threads = args.threads
    if args.company_support:
        config.salary_support.company_support = args.company_support
    count = len(support_scenarios(config))
    tests = count * len(config.salary_support.salaries)
    print(f"{count:,} life configurations x {len(config.salary_support.salaries)} fixed salaries x {config.paths:,} paths = {tests*config.paths:,} path simulations.", flush=True)
    if not args.dry_run:
        run_salary_support(config, Path(args.output), args.resume)


if __name__ == "__main__":
    main()
