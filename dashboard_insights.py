"""Cached aggregate insights, with no simulations inside the Streamlit rerun loop."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from display_names import career_group, display_name, option_name
from finance_sim.salary_support import support_statistics


def risk_trends(summary: pd.DataFrame, dimension: str) -> pd.DataFrame:
    """Descriptive trends over conditional choices, not independent pooled trials."""
    return summary.groupby(dimension, dropna=False).agg(
        average_risk=("distress_probability", "mean"),
        p10_configuration_risk=("distress_probability", lambda s: s.quantile(.1)),
        p90_configuration_risk=("distress_probability", lambda s: s.quantile(.9)),
        configurations=("scenario_id", "size"),
        depletion=("liquid_depletion_probability", "mean"),
        unpaid_bills=("shortfall_probability", "mean"),
    ).reset_index()


@st.fragment
def show_risk_trends(summary: pd.DataFrame) -> None:
    with st.expander('What worked and failed in the saved runs?', expanded=True):
        st.caption('Calculated directly from the saved simulations and your current filters. Changing this view does not run new simulations.')
        summary = summary.assign(career=summary.career.map(career_group)) if 'career' in summary else summary
        dimension = st.selectbox('Explore a trend', [c for c in ['location', 'children', 'stop_after_birth', 'macro', 'career', 'housing', 'career_year', 'move_year'] if c in summary], format_func=display_name)
        trend = risk_trends(summary, dimension)
        labels = [option_name(dimension, v) for v in trend[dimension]]
        fig = go.Figure(go.Bar(x=labels, y=trend.average_risk, name='Average conditional risk', marker_color='#b38335',
                              customdata=trend.configurations, hovertemplate='%{x}<br>Average risk %{y:.1%}<br>%{customdata} configurations<extra></extra>'))
        fig.add_trace(go.Scatter(x=labels, y=trend.p10_configuration_risk, mode='markers', name='10th percentile across configurations', marker=dict(color='#397c69', symbol='triangle-up', size=9)))
        fig.add_trace(go.Scatter(x=labels, y=trend.p90_configuration_risk, mode='markers', name='90th percentile across configurations', marker=dict(color='#233d38', symbol='triangle-down', size=9)))
        fig.update_layout(height=360, margin=dict(l=10, r=10, t=20, b=70), template='plotly_white',
                          yaxis=dict(title='Depleted funds or unpaid bills', tickformat='.0%', range=[0, 1.04]), legend=dict(orientation='h', y=-.35))
        st.plotly_chart(fig, use_container_width=True, config={'displaylogo': False})
        worst = trend.loc[trend.average_risk.idxmax()]
        best = trend.loc[trend.average_risk.idxmin()]
        st.write(f"Across the selected configurations, average modeled risk ranges from {best.average_risk:.1%} for **{option_name(dimension, best[dimension])}** to {worst.average_risk:.1%} for **{option_name(dimension, worst[dimension])}**.")
        st.caption('Each configuration receives equal weight. Marks show variation between configurations, not confidence intervals. Other choices may differ between groups, so these trends describe the tested grid and do not establish causal effects or real-world probabilities.')


def latest_support(outputs: Path) -> Path | None:
    candidates = []
    for path in outputs.glob('*/salary_support.json'):
        try:
            metadata = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        candidates.append((metadata.get('status') == 'complete', path.stat().st_mtime_ns, path.parent))
    return max(candidates)[2] if candidates else None


@st.cache_data(show_spinner=False, max_entries=3)
def load_support(folder: str, fingerprint: str, completed: int) -> pd.DataFrame:
    path = Path(folder)
    metadata = json.loads((path / 'salary_support.json').read_text())
    frames = [pd.read_parquet(path / part['paths']) for part in metadata['parts']]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


@st.cache_data(show_spinner=False, max_entries=8)
def support_table(folder: str, fingerprint: str, completed: int, age: float, budget: float,
                  requirements: tuple[str, ...] = ('retirement', 'bills', 'reserve'), minimum_paths: int = 32) -> pd.DataFrame:
    return support_statistics(load_support(folder, fingerprint, completed), age, budget, requirements, minimum_paths)


@st.fragment
def show_salary_support(outputs: Path) -> None:
    with st.expander('Salary goal without a Synthyra windfall', expanded=False):
        folder = latest_support(outputs)
        if folder is None:
            st.info('The trends above analyze existing outcomes. For an additional controlled comparison, this optional run holds salary fixed and forces all Synthyra payments to zero across the configured life choices.')
            st.code('python salary_support.py --config config.py --output outputs/salary-support')
            return
        candidates = {}
        for path in {*outputs.glob('*/salary_support.json'), folder / 'salary_support.json'}:
            try:
                info = json.loads(path.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            mode = info.get('company_support', 'none')
            rank = ({'lifecycle_v3': 2, 'lifecycle_v2': 1}.get(info.get('retirement_budget_model'), 0), info['status'] == 'complete', path.stat().st_mtime_ns)
            if mode not in candidates or rank > candidates[mode][0]:
                candidates[mode] = (rank, path.parent)
        mode = st.selectbox('Synthyra downside case', sorted(candidates, key=lambda v: v != 'none'),
                            format_func=lambda v: {'none': 'Synthyra pays nothing', 'darpa_only': 'DARPA only, then outside employment'}.get(v, display_name(v)))
        folder = candidates[mode][1]
        metadata = json.loads((folder / 'salary_support.json').read_text())
        lifecycle = metadata.get('retirement_budget_model') in ('lifecycle_v2', 'lifecycle_v3')
        if metadata.get('retirement_budget_model') == 'lifecycle_v2':
            st.warning('Superseded retirement calculation: these results can overstate saving capacity when bills or taxes remain unpaid. Use a lifecycle_v3 run for salary and retirement decisions.')
        if not lifecycle:
            st.warning('Legacy retirement projection: these saved results keep child and mortgage costs indefinitely. Updated lifecycle results will replace this view when available.')
        complete = metadata['status'] == 'complete'
        if not complete:
            st.info(f"Salary-only analysis is running: {metadata['completed_salary_configurations']:,}/{metadata['expected_salary_configurations']:,} salary/configuration tests saved. Thresholds remain withheld until coverage is complete.")
        st.caption('Independent of the main chart filters. These controlled salary runs test their own saved life choices. Changing requirements only reanalyzes saved results.')
        coverage = metadata['coverage']
        st.caption(f"{metadata['expected_configurations']:,} life configurations · {len(coverage['locations'])} locations · {len(coverage['macros'])} economic narratives · {metadata['paths_per_configuration']:,} independent paths. Only the configured birth schedules and marriage year are tested.")
        if mode == 'darpa_only':
            st.write('Only the DARPA award supports Synthyra compensation. The UD/Synthyra split ends when the award period ends, then the tested outside salary begins. No commercial revenue, follow-on funding, distributions or exit proceeds are assumed.')
        else:
            st.write('Zero Synthyra salary, distributions and exit proceeds. The tested outside salary begins in the saved start year.')
        age_options = sorted({60, 65, int(metadata['retirement_age'])})
        age = st.selectbox('Retirement age', age_options, index=age_options.index(int(metadata['retirement_age'])),
                           key='salary_support_retirement_age',
                           help='Both adults retire by this Logan age. Recalculates from saved retirement ages without new simulations.')
        labels = {'retirement': f'Retire by {age}', 'bills': 'Pay all bills', 'reserve': f"Keep a {metadata['reserve_months']:g}-month reserve"}
        requirements = tuple(st.pills('Requirements', list(labels), selection_mode='multi', default=list(labels),
                                     format_func=labels.get, key='salary_support_requirements',
                                     help='Select any combination. Every selected requirement must pass on a path. Bills also requires avoiding depleted liquid funds. Reserves are checked after Amanda stops work.'))
        if not requirements:
            st.info('Select at least one requirement to see the salary goal and trends.')
            return
        st.caption('Selected requirements must pass together. Bills includes avoiding exhausted liquid funds; the reserve is required after Amanda stops working. Retirement means both adults retire by Logan\'s target age.')
        if not metadata['parts']:
            return
        stats = support_table(str(folder), metadata['fingerprint'], metadata['completed_salary_configurations'], age, metadata['risk_budget'], requirements, metadata.get('minimum_paths', 32))
        overall = stats[(stats.dimension == 'overall') & (stats.choice == 'all')].sort_values('salary_2026')
        threshold = overall.supported_salary_2026.iloc[0] if complete else np.nan
        point_pass = overall.success_probability.ge(1-metadata['risk_budget'])
        point_candidates = overall.loc[point_pass.iloc[::-1].cummin().iloc[::-1]]
        sampling_limited = complete and pd.isna(threshold) and not point_candidates.empty
        first, second, third = st.columns([2, 1, 1])
        first.metric('Lowest supported tested salary', f'${threshold:,.0f}' if pd.notna(threshold) else ('More paths needed' if sampling_limited else ('Not reached in grid' if complete else 'Computing')))
        second.metric('Required modeled success', f"{1-metadata['risk_budget']:.0%}")
        third.metric('Requirements selected', f'{len(requirements)} of 3')
        st.caption('Starting annual employee salary in 2026 dollars. A threshold requires the simultaneous 95% Monte Carlo lower bound to pass the target at that salary and every higher tested salary. Untested salary points are not interpolated.')
        if sampling_limited:
            candidate = point_candidates.iloc[0]
            st.info(f"The point estimate reaches the target at ${candidate.salary_2026:,.0f}: {candidate.success_probability:.1%} estimated success. The simultaneous lower bound is only {candidate.success_lower_simultaneous_95:.1%} with {int(candidate.independent_paths):,} independent paths. This is insufficient sampling precision to establish the requested threshold, not evidence that the salary cannot work.")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=overall.salary_2026, y=overall.success_probability, mode='lines+markers', name='Selected requirements: estimated success', line=dict(color='#397c69', width=3)))
        fig.add_trace(go.Scatter(x=overall.salary_2026, y=overall.success_lower_simultaneous_95, mode='lines', name='Simultaneous 95% lower bound', line=dict(color='#b38335', dash='dot')))
        fig.add_hline(y=1-metadata['risk_budget'], line_dash='dash', line_color='#8a9590')
        fig.update_layout(height=340, template='plotly_white', margin=dict(l=10, r=10, t=15, b=65),
                          xaxis=dict(title='Starting salary, 2026 dollars', tickformat='$~s'),
                          yaxis=dict(title='All tested choices succeed', tickformat='.0%', range=[0, 1.04]), legend=dict(orientation='h', y=-.3))
        st.plotly_chart(fig, use_container_width=True, config={'displaylogo': False})
        chosen = overall.loc[overall.salary_2026 == threshold].iloc[0] if pd.notna(threshold) else overall.iloc[-1]
        component_risks = {'bills': chosen.depletion_probability, 'reserve': chosen.reserve_failure_probability,
                           'retirement': chosen.retirement_failure_probability}
        components = {labels[key]: component_risks[key] for key in requirements}
        limiting = max(components, key=components.get)
        if components[limiting] > 0:
            st.write(f"At **${chosen.salary_2026:,.0f}**, the most common remaining failed requirement is **{limiting}** ({components[limiting]:.1%} of paired paths). Requirements can fail together, so their probabilities are not additive.")
        else:
            st.write(f"At **${chosen.salary_2026:,.0f}**, no failures were observed in the saved paired paths. The confidence bound still accounts for finite sampling.")
        if 'bills' in requirements and chosen.pre_salary_failure_probability > 0:
            st.info(f"In {chosen.pre_salary_failure_probability:.1%} of paired paths, at least one tested life configuration ran out of funds or left bills unpaid before the outside salary began. A later salary increase cannot repair that earlier failure. Use the move-year breakdown to see the timing constraint.")
        timing = stats[(stats.dimension == 'move_year') & (stats.choice != 'not_applicable') & (stats.salary_2026 == chosen.salary_2026)]
        if 'reserve' in requirements and len(timing) > 1 and timing.reserve_failure_probability.max() - timing.reserve_failure_probability.min() > .05:
            best = timing.loc[timing.reserve_failure_probability.idxmin()]
            worst = timing.loc[timing.reserve_failure_probability.idxmax()]
            st.write(f"**Move-timing trend at ${chosen.salary_2026:,.0f}:** modeled reserve-failure risk is {worst.reserve_failure_probability:.1%} across the {worst.choice} move cases versus {best.reserve_failure_probability:.1%} across the {best.choice} move cases, testing every other configured choice within each group.")
        if 'retirement' in requirements:
            st.caption('Beyond 2036, retirement uses a separate long-term return model and inferred saving capacity. It does not extend each economic narrative through age 100. Projected deficit recovery is approximate; retirement ages on deficit paths remain provisional. Housing, benefits and taxes remain planning assumptions.')
            if lifecycle:
                st.caption('Child support ends after college, mortgage payments end at payoff, and rent or homeowner operating costs continue. Released spending increases projected savings. Remaining temporary obligations are reserved separately at zero real return; the ongoing adult budget uses stochastic portfolio returns.')
            else:
                st.caption('This older retirement projection keeps final family/housing costs indefinitely. Use updated lifecycle results for the revised salary goal.')
        dimension = st.selectbox('Break down salary requirements by', ['location', 'children', 'macro', 'stop_after_birth', 'move_year', 'housing'], format_func=display_name)
        detail = stats[stats.dimension == dimension]
        trends = go.Figure()
        for choice, frame in detail.groupby('choice'):
            frame = frame.sort_values('salary_2026')
            trends.add_trace(go.Scatter(x=frame.salary_2026, y=frame.success_probability, mode='lines', name=option_name(dimension, choice)))
        trends.add_hline(y=1-metadata['risk_budget'], line_dash='dot', line_color='#8a9590')
        trends.update_layout(height=350, template='plotly_white', margin=dict(l=10, r=10, t=15, b=70),
                             xaxis=dict(title='Starting salary, 2026 dollars', tickformat='$~s'),
                             yaxis=dict(title='Modeled success', tickformat='.0%', range=[0, 1.04]), legend=dict(orientation='h', y=-.3))
        st.plotly_chart(trends, use_container_width=True, config={'displaylogo': False})
        st.caption('Each line tests every other configured choice within that group. These are matched salary reruns with common economic draws, not salary bins or averages of scenario percentiles. One paired path remains one independent observation even when reused across thousands of choices. The all-choices test requires every tested configuration to succeed on a path, which is stricter than an average success rate.')
        export = stats.assign(retirement_target_age=age, selected_requirements=','.join(requirements),
                              run=folder.name, coverage_complete=complete,
                              interpretation='Conditional all-choice success; main chart filters do not apply')
        if not complete:
            export['supported_salary_2026'] = np.nan
        st.download_button('Download salary analysis', export.to_csv(index=False),
                           f'salary_analysis_age{age}.csv', 'text/csv')
        if st.checkbox('Show coverage and assumptions for this question'):
            coverage = metadata['coverage']
            st.write(f"Dedicated run: {folder.name}. {metadata['expected_configurations']:,} life configurations, {len(metadata['salaries_2026'])} fixed salaries, {metadata['paths_per_configuration']:,} independent paths each. This section is independent of the main chart filters.")
            st.write(f"Outside salary begins in {coverage['salary_start_year']}-{coverage.get('salary_start_month', 1):02d}; marriage is modeled in {coverage['marriage_year'] or 'no year (unmarried)'}. Move years: {', '.join(map(str, coverage['move_years']))}. Birth schedules: {'; '.join(', '.join(map(str, schedule)) for schedule in coverage['birth_schedules'])}.")
            st.write(f"{len(coverage['locations'])} locations and {len(coverage['macros'])} economic scenarios. Family/employee benefits follow the saved assumptions. The {metadata['retirement_funding_target']:.0%} retirement-funding target through age {metadata['retirement_life_expectancy']} is distinct from the outer salary-support probability.")
            st.caption(metadata['notice'])
            if lifecycle:
                st.write(f"Child support ends at age {coverage['child_support_end_age']}. Parents fund {coverage['college_parent_share']:.0%} of the ${coverage['college_annual_cost']:,.0f} annual college cost in 2026 dollars. Mortgage term: {coverage['mortgage_years']} years from actual purchase. Rent continues for paths without a home purchase.")
            st.caption('No conclusion covers untested birth dates, more children than configured, every possible future economy, or uncertainty in all cost/tax assumptions. Beyond 2036, retirement follows the configured long-term return and saving projection.')
