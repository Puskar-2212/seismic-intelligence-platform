from streamlit.testing.v1 import AppTest

from dashboard.pages import PAGES
from seismic.paths import PROJECT_ROOT


def test_all_dashboard_pages_render():
    app = AppTest.from_file(str(PROJECT_ROOT / "dashboard" / "app.py"), default_timeout=30).run()
    assert not app.exception
    assert app.metric[0].value == "70"
    for page in PAGES:
        app.sidebar.radio[0].set_value(page).run()
        assert not app.exception, page


def test_filters_update_kpis_and_handle_empty_results():
    app = AppTest.from_file(str(PROJECT_ROOT / "dashboard" / "app.py"), default_timeout=30).run()
    app.sidebar.slider[0].set_value((3.0, 3.8)).run()
    assert not app.exception
    assert int(app.metric[0].value) < 70
    app.sidebar.text_input[0].set_value("no-such-location").run()
    assert not app.exception
    assert any("No events match" in message.value for message in app.info)


def test_model_page_accepts_matching_results_and_rejects_stale(events, tmp_path, monkeypatch):
    import dashboard.pages as pages
    from seismic.modeling import evaluate_models

    monkeypatch.setattr(pages, "ARTIFACTS", tmp_path)
    evaluate_models(events, tmp_path / "model", include_lstm=False)
    app = AppTest.from_file(str(PROJECT_ROOT / "dashboard" / "app.py"), default_timeout=30).run()
    app.sidebar.radio[0].set_value("Forecasting / Model Evaluation").run()
    assert not app.exception
    assert len(app.dataframe) == 1
    changed = events.copy()
    changed.loc[0, "mag"] += 0.1
    evaluate_models(changed, tmp_path / "model", include_lstm=False)
    app.run()
    assert not app.exception
    assert any("different catalogue" in message.value for message in app.warning)


def test_optional_fields_and_single_day_catalogue_render(raw, monkeypatch):
    import seismic.data as data_module

    fixture = raw.iloc[:3][["date_time", "mag"]].copy()
    catalogue = data_module.normalize_catalogue(fixture)
    monkeypatch.setattr(data_module, "load_catalogue", lambda *args: catalogue)
    app = AppTest.from_file(str(PROJECT_ROOT / "dashboard" / "app.py"), default_timeout=30).run()
    for page in PAGES:
        app.sidebar.radio[0].set_value(page).run()
        assert not app.exception, page


def test_invalid_input_shows_error(monkeypatch):
    import seismic.data as data_module

    def fail(*args):
        raise ValueError("Required columns missing: date_time, mag")

    monkeypatch.setattr(data_module, "load_catalogue", fail)
    app = AppTest.from_file(str(PROJECT_ROOT / "dashboard" / "app.py"), default_timeout=30).run()
    assert not app.exception
    assert "Required columns" in app.error[0].value
