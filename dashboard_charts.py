"""Efficient overview charts over every selected scenario's monthly medians."""

import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pyarrow.dataset as ds

from pathlib import Path

from display_names import career_group


def read_trajectories(folder: Path, metric: str) -> pd.DataFrame:
    """Project one metric from committed partitions instead of reading all bands."""
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    if manifest["schema_version"] != 1:
        raise ValueError("Unsupported run schema")
    files = [folder / "monthly" / part["file"] for part in manifest["parts"]]
    if not files:
        return pd.DataFrame(columns=["scenario_id", "date", "q500"])
    return ds.dataset(files, format="parquet").to_table(
        columns=["scenario_id", "date", "q025", "q100", "q250", "q500", "q750", "q900", "q975"], filter=ds.field("metric") == metric,
    ).to_pandas()


def all_scenario_chart(trajectories: pd.DataFrame, selected: pd.DataFrame,
                       interval: str = "None", scale: str = "Linear") -> go.Figure:
    """Use a WebGL trace per career, with gaps separating individual scenarios."""
    filtered = trajectories[trajectories.scenario_id.isin(selected.scenario_id)]
    matrix = filtered.pivot(index="scenario_id", columns="date", values="q500").sort_index(axis=1)
    dates = pd.DatetimeIndex(matrix.columns).as_unit("ms").asi8.astype(float)  # (months,)
    fig = go.Figure()
    groups = selected.assign(career_family=selected.career.map(career_group))
    for career, scenarios in groups.groupby("career_family", sort=False):
        ids = [sid for sid in scenarios.scenario_id if sid in matrix.index]
        values = matrix.loc[ids].to_numpy()  # (scenarios, months)
        if not len(values):
            continue
        # NaNs prevent a line from connecting one scenario's end to the next start.
        x = np.tile(np.append(dates, np.nan), len(ids))  # (scenarios * (months+1),)
        y = np.column_stack((values, np.full(len(ids), np.nan))).ravel().astype(np.float32)  # (scenarios * (months+1),)
        fig.add_trace(go.Scattergl(
            x=x, y=y, name=career, legendgroup=career,
            mode="lines+markers", marker=dict(size=3), connectgaps=False, line=dict(width=1),
            meta=dict(scenario_ids=ids, points_per_scenario=len(dates)+1),
            opacity=.3 if len(selected) > 100 else .65,
            hovertemplate="%{x|%b %Y}<br>%{y:$,.0f}<extra>%{fullData.name}</extra>",
        ))
    fig.update_layout(
        height=560, title=f"{len(matrix):,} scenario trajectories",
        xaxis=dict(type="date", title="Year"), yaxis_title="Nominal USD",
        legend=dict(orientation="h", y=-.18, title="Logan's career", font=dict(size=11)),
        meta=dict(scenario_count=len(matrix)), clickmode="event+select", dragmode="zoom",
    )
    if interval != "None":
        lower, upper = {"50%": ("q250", "q750"), "80%": ("q100", "q900"), "95%": ("q025", "q975")}[interval]
        envelope = filtered.groupby("date").agg(low=(lower, "min"), high=(upper, "max"))
        fig.add_trace(go.Scatter(x=envelope.index, y=envelope.low, mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=envelope.index, y=envelope.high, mode="lines", line=dict(width=0), fill="tonexty",
                                fillcolor="rgba(57,124,105,.12)", name=f"{interval} scenario range envelope", hoverinfo="skip"))
    scale_money_axis(fig, scale)
    return fig


def scale_money_axis(fig: go.Figure, scale: str = "Linear") -> None:
    """Signed compression preserves zero and negatives, without clipping outliers."""
    arrays = [np.asarray(trace.y, dtype=float) for trace in fig.data if trace.y is not None]
    finite = np.concatenate(arrays) if arrays else np.array([0.])
    finite = finite[np.isfinite(finite)]
    low, high = (min(0., finite.min()), max(0., finite.max())) if len(finite) else (0., 0.)
    if scale == "Balanced":
        # Linear around zero, logarithmic for large magnitudes. $10k is a display
        # scale only, never a financial assumption or clipped lower limit.
        knee = 10_000.
        for trace in fig.data:
            raw = np.asarray(trace.y, dtype=float)
            trace.customdata = raw.astype(np.float32)
            trace.y = np.arcsinh(raw / knee).astype(np.float32)
            if trace.hoverinfo != "skip":
                trace.hovertemplate = "%{x|%b %Y}<br>%{customdata:$,.0f}<extra>%{fullData.name}</extra>"
        magnitudes = [1e4, 5e4, 2e5, 1e6, 5e6, 2e7, 1e8, 5e8, 2e9, 1e10, 5e10, 2e11]
        ticks = [v for v in sorted([-v for v in magnitudes] + [0.] + magnitudes) if low <= v <= high]
        def label(value: float) -> str:
            absolute = abs(value)
            unit, divisor = ("B", 1e9) if absolute >= 1e9 else (("M", 1e6) if absolute >= 1e6 else ("k", 1e3))
            return ("−" if value < 0 else "") + (f"${absolute/divisor:g}{unit}" if value else "$0")
        low, high = np.arcsinh(low / knee), np.arcsinh(high / knee)
        fig.update_yaxes(tickmode="array", tickvals=np.arcsinh(np.asarray(ticks)/knee), ticktext=[label(v) for v in ticks])
    else:
        fig.update_yaxes(tickformat="$~s", nticks=8, separatethousands=True)
    padding = max((high - low) * .06, .1 if scale == "Balanced" else 1.)
    fig.update_yaxes(range=[low-padding, high+padding], automargin=True, zeroline=True, zerolinecolor="#9eaca3")
    fig.update_xaxes(automargin=True, dtick="M12", tickformat="%Y")


def clicked_scenario(event: dict, fig: go.Figure) -> str | None:
    points = event.get("selection", {}).get("points", [])
    if not points:
        return None
    point = points[-1]
    trace = fig.data[int(point["curve_number"])]
    if not trace.meta or "scenario_ids" not in trace.meta:
        return None
    index = int(point["point_index"]) // trace.meta["points_per_scenario"]
    return trace.meta["scenario_ids"][index]
