from collections.abc import Iterable

import dash_bootstrap_components as dbc

import app as app_module
from app import STRATEGIES, STRATEGY_INFO, update_graph, validate_inputs


def _walk_components(component):
    if component is None:
        return
    if isinstance(component, (str, int, float, bool)):
        return
    if isinstance(component, dict):
        for value in component.values():
            yield from _walk_components(value)
        return
    if isinstance(component, Iterable) and not hasattr(component, "children"):
        for item in component:
            yield from _walk_components(item)
        return

    yield component
    children = getattr(component, "children", None)
    if children is not None:
        yield from _walk_components(children)


def test_dualpipe_v_is_available_to_the_web_ui():
    assert "dualpipe_v" in STRATEGIES
    assert STRATEGY_INFO["dualpipe_v"]["name"] == "DualPipeV"


def test_dualpipe_v_uses_split_backward_validation():
    result = validate_inputs(
        4,
        8,
        10,
        0.0,
        1.0,
        None,
        None,
        1.0,
        None,
        None,
        ["dualpipe_v"],
    )

    (
        generate_disabled,
        _,
        _,
        _,
        _,
        _,
        backward_invalid,
        backward_d_invalid,
        backward_w_invalid,
        *_,
    ) = result
    assert generate_disabled is True
    assert backward_invalid is False
    assert backward_d_invalid is True
    assert backward_w_invalid is False


def test_update_graph_renders_dualpipe_v_schedule_without_error_toast():
    output, toasts = update_graph(
        1,
        4,
        8,
        10,
        0.0,
        1.0,
        None,
        1.0,
        1.0,
        None,
        None,
        ["dualpipe_v"],
        [],
        [],
    )

    toast_headers = [getattr(toast, "header", "") for toast in toasts]
    assert not any(str(header).startswith("Error:") for header in toast_headers)
    tabs = [
        component
        for component in _walk_components(output)
        if isinstance(component, dbc.Tab)
    ]
    assert any(tab.label == "DualPipeV" for tab in tabs)


def test_run_app_uses_current_dash_server_api(monkeypatch):
    recorded = {}

    def fake_run(**kwargs):
        recorded.update(kwargs)

    monkeypatch.setattr(app_module.app, "run", fake_run)

    app_module.run_app()

    assert recorded == {"debug": False, "host": "0.0.0.0", "port": 7860}
