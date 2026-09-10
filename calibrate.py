"""Download public monthly economic anchors and create an inspectable history file.

Run this explicitly. Simulations never refresh external data implicitly.
"""

import argparse
import hashlib
import io
import json
import urllib.request

import pandas as pd

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path


SERIES = {"SP500": "S&P 500 price index; excludes dividends", "CPIAUCSL": "US CPI seasonally adjusted",
          "CSUSHPINSA": "US national Case-Shiller house-price index, not neighborhood prices",
          "TB3MS": "3-month Treasury discount yield proxy for cash, annual percent"}


def fetch_series(name: str, start: str, end: str) -> tuple[str, bytes, str]:
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={name}&cosd={start}&coed={end}"
    request = urllib.request.Request(url, headers={"User-Agent": "HouseholdFinanceResearch/0.1"})
    with urllib.request.urlopen(request, timeout=45) as response:
        return name, response.read(), url


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="data/calibration")
    parser.add_argument("--start", default="2016-01-01")
    parser.add_argument("--as-of", default="2026-09-10")
    parser.add_argument("--dividend-yield", type=float, default=.008, help="Explicit annual dividend proxy added to price returns")
    args = parser.parse_args()
    folder = Path(args.output)
    folder.mkdir(parents=True, exist_ok=True)
    series, sources = {}, []
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = pool.map(lambda name: fetch_series(name, args.start, args.as_of), SERIES)
        for name, content, url in results:
            (folder / f"{name}.csv").write_bytes(content)
            frame = pd.read_csv(io.BytesIO(content), na_values=".")
            frame.columns = ["date", name]
            frame.date = pd.to_datetime(frame.date)
            frame = frame[frame.date < pd.Timestamp(args.as_of)].set_index("date")
            series[name] = frame[name].resample("MS").last()
            sources.append({"series": name, "description": SERIES[name], "url": url,
                            "sha256": hashlib.sha256(content).hexdigest()})
    levels = pd.concat(series, axis=1)
    # Only full months are included. No interpolation or filling missing observations.
    levels = levels[levels.index < pd.Timestamp(args.as_of).replace(day=1)]
    history = pd.DataFrame({"equity_return": levels.SP500.pct_change(fill_method=None) + args.dividend_yield/12,
                            "inflation": levels.CPIAUCSL.pct_change(fill_method=None),
                            "home_return": levels.CSUSHPINSA.pct_change(fill_method=None),
                            "cash_rate": levels.TB3MS / 100}).dropna()
    if len(history) < 24:
        raise ValueError("Insufficient aligned history; at least 24 complete months required")
    missing_months = pd.date_range(history.index.min(), history.index.max(), freq="MS").difference(history.index)
    history.index.name = "date"
    history.to_csv(folder / "history.csv")
    metadata = {"downloaded_utc": datetime.now(timezone.utc).isoformat(), "as_of": args.as_of,
                "start": str(history.index.min().date()), "end": str(history.index.max().date()), "months": len(history),
                "excluded_months": [str(date.date()) for date in missing_months],
                "dividend_proxy": args.dividend_yield, "sources": sources,
                "limitations": ["SP500 is a price index with a constant dividend proxy, not a total-return index or VOOG history.",
                                "Treasury discount yield is a cash-rate proxy, not a guaranteed bank APY.",
                                "Latest revised historical data, not vintage data known at each historic date.",
                                "National housing returns do not establish local price levels or local volatility."],
                "annualized_arithmetic_means": (history.mean()*pd.Series({"equity_return":12,"inflation":12,"home_return":12,"cash_rate":1})).to_dict(),
                "monthly_correlation": history.corr().to_dict()}
    (folder / "sources.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved {len(history)} aligned months to {folder / 'history.csv'}")


if __name__ == "__main__":
    main()
