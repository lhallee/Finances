"""Exercise filtering, uncertainty controls, point inspection and partial coverage."""
import json
import pytest
from pathlib import Path
from streamlit.testing.v1 import AppTest
from finance_sim.configuration import RunConfig
from finance_sim.scenarios import Scenario
from finance_sim.workflow import execute


@pytest.fixture(autouse=True)
def isolate_salary_artifacts(monkeypatch):
    monkeypatch.setattr('dashboard_insights.latest_support', lambda outputs: None)


def test_app_filters_ranges_and_details(tmp_path):
    folder = tmp_path / 'run'
    config = RunConfig(paths=2, threads=2, generate_reports=False)
    cases = [Scenario(births=(2029,)), Scenario(births=(2029,), stop_after_birth=1)]
    execute(config, folder, selected=cases)
    original = (folder / 'manifest.json').read_bytes()
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / 'app.py'))
    app.run(timeout=60)
    app.text_input[0].set_value(str(folder)).run(timeout=60)
    assert not app.exception
    assert not app.tabs
    assert len(app.get('plotly_chart')) == 2
    assert any('2 / 2 scenarios' in item.value for item in app.caption)
    next(item for item in app.multiselect if item.label == "Amanda's employment").set_value([1]).run(timeout=60)
    assert not app.exception
    assert any('1 / 2 scenarios' in item.value for item in app.caption)
    next(item for item in app.selectbox if item.label == 'Predictive range').set_value('50%').run(timeout=60)
    next(item for item in app.selectbox if item.label == 'Dollar scale').set_value('Linear').run(timeout=60)
    assert not app.exception
    app.session_state[f'clicked-{folder.resolve()}'] = cases[1].id
    app.run(timeout=60)
    assert len(app.metric) == 3
    assert any('Selected' not in h.value and 'Synthyra' in h.value for h in app.subheader)
    assert len(app.get('download_button')) == 1
    next(item for item in app.multiselect if item.label == "Amanda's employment").set_value([]).run(timeout=60)
    assert any('No saved scenarios match' in item.value for item in app.info)
    assert (folder / 'manifest.json').read_bytes() == original


def test_app_partial_run(tmp_path):
    folder = tmp_path / 'partial'
    config = RunConfig(paths=2, threads=2, batch_size=1, generate_reports=False)
    execute(config, folder, selected=[Scenario(), Scenario(births=(2029,))], maximum_batches=1)
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / 'app.py'))
    app.run(timeout=60)
    app.text_input[0].set_value(str(folder)).run(timeout=60)
    assert not app.exception
    assert any('incomplete' in warning.value for warning in app.warning)
