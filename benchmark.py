"""Benchmark compiled household runs and optional GPU economic accumulation."""

import argparse
import dataclasses
import json
import time
import subprocess
import sys

import numba
import numpy as np

from pathlib import Path

from finance_sim.configuration import load_config
from finance_sim.economics import economic_paths
from finance_sim.engine import simulate
from finance_sim.scenarios import Scenario


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.py")
    parser.add_argument("--paths", type=int, default=256)
    parser.add_argument("--output", default="outputs/benchmark.json")
    args = parser.parse_args()
    config = load_config(args.config)
    config.paths = args.paths
    numba.set_num_threads(min(config.threads, numba.config.NUMBA_NUM_THREADS))
    start = time.perf_counter()
    draws = economic_paths(config, "baseline")  # (p,t,8)
    simulate(config, Scenario(), draws)
    cold_seconds = time.perf_counter() - start
    start = time.perf_counter()
    household, company = simulate(config, Scenario(), draws)
    warm_seconds = time.perf_counter() - start
    report = {"paths": args.paths, "months": draws.shape[1], "cpu_threads": numba.get_num_threads(),
              "first_call_seconds": cold_seconds, "warm_simulation_seconds": warm_seconds,
              "backend": "numba_cpu", "gpu_scope": "economic accumulation only; household/tax kernels remain compiled CPU"}
    # Driver initialization can hang on a misconfigured local GPU. Isolate it
    # with a bounded probe so CPU measurements and the application remain usable.
    probe = r"""
import json, time
import numpy as np
import cupy as cp
returns = np.random.default_rng(1).normal(0, .05, (8192, 123))
start=time.perf_counter()
reference=np.cumprod(1+returns, axis=1)
cpu=time.perf_counter()-start
cp.cumprod(cp.ones((2,4), dtype=cp.float64),axis=1)
cp.cuda.Stream.null.synchronize()
start=time.perf_counter()
actual=cp.asnumpy(cp.cumprod(1+cp.asarray(returns),axis=1))
cp.cuda.Stream.null.synchronize()
gpu=time.perf_counter()-start
np.testing.assert_allclose(actual,reference,rtol=1e-12,atol=1e-12)
print(json.dumps(dict(gpu_available=True,gpu_stage_seconds=gpu,cpu_stage_seconds=cpu,gpu_stage_parity=True)))
"""
    try:
        result = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, timeout=20)
        if result.returncode:
            report.update(gpu_available=False, gpu_error=result.stderr[-1500:])
        else:
            report.update(json.loads(result.stdout))
    except subprocess.TimeoutExpired:
        report.update(gpu_available=None, gpu_error="GPU probe exceeded 20 seconds; no GPU parity or performance conclusion")
    report["decision"] = "Use compiled CPU. Full household/tax GPU kernel is not implemented or validated."
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
