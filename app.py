"""Fast single-page explorer of every saved scenario and its predictive ranges."""

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
import streamlit as st

from dashboard_insights import show_risk_trends, show_salary_support
from dashboard_charts import all_scenario_chart, clicked_scenario, read_trajectories
from display_names import career_group, display_frame, display_name, option_name
from finance_sim.configuration import load_config
from finance_sim.scenarios import scenario_from_record
from finance_sim.storage import read_table
from scenario_details import event_timeline
from showcase import latest_report


def arguments():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--output')
    args = parser.parse_known_args()[0]
    if args.output is None:
        latest = latest_report(Path(__file__).parent / 'outputs')
        args.output = str(latest.folder) if latest else 'outputs/demo'
    return args


@st.cache_data(show_spinner=False, max_entries=4)
def load_summary(path: str, revision: int) -> pd.DataFrame:
    return read_table(Path(path), 'summary')


@st.cache_resource(show_spinner=False, max_entries=2)
def load_trajectories(path: str, revision: int, metric: str) -> pd.DataFrame:
    # Shared read-only frame avoids copying the entire catalog on every click.
    return read_trajectories(Path(path), metric)


@st.cache_resource(show_spinner=False, max_entries=3)
def overview_figure(path: str, revision: int, metric: str, choices: tuple, interval: str, scale: str):
    selected = pd.DataFrame(choices, columns=['scenario_id', 'career'])
    fig = all_scenario_chart(load_trajectories(path, revision, metric), selected, interval, scale)
    manifest = json.loads((Path(path) / 'manifest.json').read_text())
    fig.update_yaxes(title=manifest.get('dollar_basis', 'Nominal USD (legacy run)'))
    fig.update_layout(template='plotly_white', paper_bgcolor='#f8faf7', plot_bgcolor='#f8faf7',
                      font=dict(family='Arial', color='#263d36'), margin=dict(l=10, r=15, t=35, b=75),
                      colorway=['#397c69', '#b38335', '#85635c', '#497d96', '#786898', '#8e9946'])
    return fig


def inspect_scenario(record: pd.Series, folder: Path) -> None:
    st.subheader(f'{display_name(record.career)} · {display_name(record.location)}')
    st.caption(f'Scenario {record.scenario_id} · Configured choices for this median trajectory')
    if 'median_minimum_retirement_age' in record:
        age = record.median_minimum_retirement_age
        first, second, third = st.columns(3)
        first.metric('Median earliest retirement age', f'{age:.1f}' if pd.notna(age) else f'Not reached by {int(record.retirement_maximum_age)}')
        second.metric('Ready within simulation', f'{record.retirement_within_horizon_probability:.0%}')
        third.metric('Ready by maximum age', f'{record.retirement_reached_probability:.0%}')
        st.caption('Retirement ages refer to Logan, assuming both adults stop working. Ages beyond 2036 are conditional projections; unresolved paths remain in the median. Retirement assumptions are in the saved inputs.')
    fields = ('career', 'career_year', 'location', 'move_year', 'housing', 'marriage_year', 'births', 'stop_after_birth',
              'amanda_career', 'macro', 'benefits', 'company_mode', 'company_outcome', 'grants', 'grant_case', 'exit_value', 'exit_year', 'exit_type')
    left, right = st.columns(2)
    with left:
        st.dataframe(pd.DataFrame({'Option': [display_name(f) for f in fields], 'Choice': [option_name(f, record[f]) for f in fields]}), hide_index=True, use_container_width=True)
    with right:
        st.dataframe(event_timeline(scenario_from_record(record.to_dict()), load_config(folder / 'config.json')), hide_index=True, use_container_width=True)
        st.caption('The timeline shows scheduled events. Actual job gaps, purchases and payments vary between stochastic paths; a median line is not an individual path.')


def main() -> None:
    st.set_page_config(page_title='Possible futures', page_icon='🌿', layout='wide')
    st.markdown('''<style>
    .stApp {background:#f8faf7;color:#263d36}
    h1,h2,h3 {letter-spacing:-.025em}
    [data-testid="stSidebar"] {background:#edf2eb}
    .block-container {padding-top:2rem;padding-bottom:2rem}
    button:focus-visible,a:focus-visible {outline:3px solid #b38335!important}
    </style>''', unsafe_allow_html=True)
    st.title('Possible futures')
    st.caption('Household finances · 2027–2036 · Every scenario, one view')
    with st.sidebar.expander('Saved results'):
        folder = Path(st.text_input('Output folder', value=arguments().output)).expanduser().resolve()
        if st.button('Refresh results'):
            st.cache_data.clear()
            st.cache_resource.clear()
    if not (folder / 'manifest.json').exists():
        st.info('Choose a simulation output folder to explore its results.')
        st.code('python simulate.py --preset demo --output outputs/demo')
        return
    manifest = json.loads((folder / 'manifest.json').read_text(encoding='utf-8'))
    revision = manifest['completed_scenarios']
    if manifest['status'] != 'complete':
        st.warning(f"This run is incomplete: {revision:,} of {manifest['expected_scenarios']:,} scenarios saved.")
    summary = load_summary(str(folder), revision)
    if summary.empty:
        st.info('Waiting for the first committed batch.')
        return
    filtered = summary
    with st.sidebar:
        st.subheader('Keep the possibilities you want')
        st.caption('Removing an option removes its trajectories immediately.')
        primary = (('career', "Logan's career"), ('location', 'Location'), ('children', 'Children'),
                   ('stop_after_birth', "Amanda's employment"), ('macro', 'Economic scenario'))
        secondary = (('housing', 'Housing'), ('benefits', 'UD benefits'), ('company_outcome', 'Company outcome'),
                     ('company_mode', 'Company model'), ('grant_case', 'Grant outcome'), ('grants', 'Funding portfolio'),
                     ('exit_value', 'Exit value'), ('marriage_year', 'Marriage year'), ('births', 'Birth years'),
                     ('amanda_career', "Amanda's career"), ('career_year', 'Career change year'), ('move_year', 'Move year'))
        for key, label in primary:
            if key == 'career':
                options = sorted(summary.career.map(career_group).unique().tolist())
                chosen = st.multiselect(label, options, default=options)
                filtered = filtered[filtered.career.map(career_group).isin(chosen)]
            else:
                options = sorted(summary[key].dropna().unique().tolist())
                chosen = st.multiselect(label, options, default=options, format_func=lambda value, field=key: option_name(field, value))
                filtered = filtered[filtered[key].isin(chosen)]
        with st.expander('More options'):
            for key, label in secondary:
                options = sorted(summary[key].dropna().unique().tolist())
                chosen = st.multiselect(label, options, default=options, format_func=lambda value, field=key: option_name(field, value))
                filtered = filtered[filtered[key].isin(chosen)]
    if filtered.empty:
        st.info('No saved scenarios match. Add an option back to see trajectories.')
        return
    first, second, third = st.columns([2, 1, 1])
    with first:
        metric = st.selectbox('Measure', ['net_worth', 'liquid_wealth', 'cash', 'retirement', 'debt', 'home_equity', 'earned_income', 'tax_expense'], format_func=display_name)
    with second:
        interval = st.selectbox('Predictive range', ['95%', '80%', '50%', 'None'])
    with third:
        scale = st.selectbox('Dollar scale', ['Balanced', 'Linear'], help='Balanced compresses large values while preserving zero and negative balances. Linear shows equal dollar distances.')
    st.caption(f"{len(filtered):,} / {len(summary):,} scenarios · {int(filtered.paths.sum()):,} stochastic paths · Lines show each scenario's median; no scenarios are sampled.")
    st.caption(f"Dollar basis: {manifest.get('dollar_basis', 'Nominal USD, legacy run. Generate new results for 2026-dollar views.')}")
    choices = tuple(filtered[['scenario_id', 'career']].itertuples(index=False, name=None))
    key = hashlib.sha256(repr((str(folder), revision, metric, choices, interval, scale)).encode()).hexdigest()[:16]
    with st.spinner('Preparing trajectories...'):
        fig = overview_figure(str(folder), revision, metric, choices, interval, scale)
    event = st.plotly_chart(fig, use_container_width=True, key=f'overview-{key}', on_select='rerun',
                           selection_mode='points', config={'displaylogo': False, 'scrollZoom': True})
    st.caption("Shading encloses every selected scenario's central predictive range. It is an envelope across conditional scenarios, not a pooled confidence interval or probability forecast. Balanced dollar spacing is nonlinear; negative outcomes stay visible. Drag to zoom, double-click to reset.")
    clicked = clicked_scenario(event, fig)
    if clicked:
        st.session_state[f'clicked-{folder}'] = clicked
    focused = st.session_state.get(f'clicked-{folder}')
    if focused in set(filtered.scenario_id):
        inspect_scenario(filtered[filtered.scenario_id == focused].iloc[0], folder)
    else:
        st.caption('Click a point on any trajectory to see its options, event timeline and retirement outlook.')
    if 'distress_probability' in filtered:
        show_risk_trends(filtered)
    show_salary_support(Path(__file__).parent / 'outputs')
    with st.expander('Results table and downloads'):
        columns = ['scenario_id', 'career', 'location', 'children', 'stop_after_birth', 'median_net_worth',
                   'median_minimum_retirement_age', 'retirement_reached_probability', 'shortfall_probability']
        st.dataframe(display_frame(filtered[[c for c in columns if c in filtered]]), hide_index=True, use_container_width=True)
        st.download_button('Download filtered results', filtered.to_csv(index=False), 'filtered_scenarios.csv', 'text/csv')
    with st.expander('Sources and assumptions'):
        st.write('Each line is conditional on its configuration. Monte Carlo ranges quantify modeled uncertainty, not uncertainty in every input assumption.')
        st.dataframe(pd.read_json(folder / 'sources.json'), hide_index=True, use_container_width=True)
        st.json(json.loads((folder / 'config.json').read_text(encoding='utf-8')), expanded=False)


if __name__ == '__main__':
    main()
