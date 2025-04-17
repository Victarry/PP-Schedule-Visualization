import dash
import dash_bootstrap_components as dbc
from dash import dcc, html, Input, Output, State, callback_context
import plotly.graph_objects as go
import webbrowser
from threading import Timer

from src.execution_model import ScheduleConfig, Schedule
from src.strategies import (
    generate_1f1b_schedule,
    generate_zero_bubble_1p_schedule,
    generate_1f1b_overlap_schedule,
    generate_1f1b_interleave_schedule,
    generate_1f1b_interleave_overlap_schedule,
    generate_dualpipe_schedule
)
from src.visualizer import convert_schedule_to_visualization_format, create_pipeline_figure

def open_browser(port):
    webbrowser.open_new(f"http://127.0.0.1:{port}")

STRATEGIES = {
    "1f1b": generate_1f1b_schedule,
    "zb1p": generate_zero_bubble_1p_schedule,
    "1f1b_overlap": generate_1f1b_overlap_schedule,
    "1f1b_interleave": generate_1f1b_interleave_schedule,
    "1f1b_interleave_overlap": generate_1f1b_interleave_overlap_schedule,
    "dualpipe": generate_dualpipe_schedule,
}

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True)
app.title = "Pipeline Parallelism Schedule Visualizer"

# Initial default values
default_values = {
    "num_devices": 4,
    "num_stages": 8,
    "num_batches": 16,
    "p2p_latency": 0.1,
    "op_time_forward": 1.0,
    "op_time_backward_d": 1.0,
    "op_time_backward_w": 1.0,
    "op_time_backward": 2.0,
    "strategy": "1f1b_interleave",
    "split_backward": False,
    "placement_strategy": "interleave"
}

# Define input groups using dbc components
basic_params_card = dbc.Card(
    dbc.CardBody([
        html.H5("Basic Parameters", className="card-title"),
        html.Div([
            dbc.Label("Number of Devices (GPUs):"),
            dbc.Input(id='num_devices', type='number', value=default_values["num_devices"], min=1, step=1),
        ], className="mb-3"),
        html.Div([
            dbc.Label("Number of Stages (Model Chunks):"),
            dbc.Input(id='num_stages', type='number', value=default_values["num_stages"], min=1, step=1),
        ], className="mb-3"),
        html.Div([
            dbc.Label("Number of Microbatches:"),
            dbc.Input(id='num_batches', type='number', value=default_values["num_batches"], min=1, step=1),
        ], className="mb-3"),
        html.Div([
            dbc.Label("P2P Latency (ms):"),
            dbc.Input(id='p2p_latency', type='number', value=default_values["p2p_latency"], min=0, step=0.01),
        ], className="mb-3"),
    ])
)

scheduling_params_card = dbc.Card(
    dbc.CardBody([
        html.H5("Scheduling Parameters", className="card-title"),
        html.Div([
            dbc.Label("Scheduling Strategies:"),
            dbc.Checklist(
                id='strategy-checklist',
                options=[{'label': k, 'value': k} for k in STRATEGIES.keys()],
                value=[default_values["strategy"]],
                inline=False,
            ),
        ], className="mb-3"),
    ])
)

timing_params_card = dbc.Card(
    dbc.CardBody([
        html.H5("Operation Timing (ms)", className="card-title"),
        html.Div([
            dbc.Label("Forward:"),
            dbc.Input(id='op_time_forward', type='number', value=default_values["op_time_forward"], min=0.01, step=0.01),
        ], className="mb-3"),
        html.Div([
            dbc.Label("Backward (Combined):"),
            dbc.Input(id='op_time_backward', type='number', value=default_values["op_time_backward"], min=0.01, step=0.01),
            dbc.FormText("Used when strategy does NOT require split backward."),
        ], className="mb-3"),
        html.Div([
            dbc.Label("Backward D (Data Grad):"),
            dbc.Input(id='op_time_backward_d', type='number', value=default_values["op_time_backward_d"], min=0.01, step=0.01),
            dbc.FormText("Used when strategy requires split backward (e.g., ZB-1P, DualPipe)."),
        ], className="mb-3"),
        html.Div([
            dbc.Label("Backward W (Weight Grad):"),
            dbc.Input(id='op_time_backward_w', type='number', value=default_values["op_time_backward_w"], min=0.01, step=0.01),
            dbc.FormText("Used when strategy requires split backward (e.g., ZB-1P, DualPipe)."),
        ], className="mb-3"),
    ])
)

# Updated app layout using dbc components and structure
app.layout = dbc.Container([
    html.H1("Pipeline Parallelism Schedule Visualizer", className="my-4 text-center"),

    dbc.Row([
        dbc.Col(basic_params_card, md=4),
        dbc.Col(scheduling_params_card, md=4),
        dbc.Col(timing_params_card, md=4),
    ]),

    dbc.Row([
        dbc.Col([
            dbc.Button('Generate Schedule', id='generate-button', n_clicks=0, color="primary", className="mt-4"),
        ], className="text-center")
    ]),

    dbc.Row([
        dbc.Col([
            dcc.Loading(
                id="loading-graph-area",
                type="circle",
                children=html.Div(id='graph-output-container', className="mt-4")
            )
        ])
    ])
], fluid=True)

@app.callback(
    Output('graph-output-container', 'children'),
    Input('generate-button', 'n_clicks'),
    State('num_devices', 'value'),
    State('num_stages', 'value'),
    State('num_batches', 'value'),
    State('p2p_latency', 'value'),
    State('op_time_forward', 'value'),
    State('op_time_backward', 'value'),
    State('op_time_backward_d', 'value'),
    State('op_time_backward_w', 'value'),
    State('strategy-checklist', 'value'),
    prevent_initial_call=True
)
def update_graph(n_clicks, num_devices, num_stages, num_batches, p2p_latency,
                 op_time_forward, op_time_backward, op_time_backward_d, op_time_backward_w,
                 selected_strategies):

    output_components = []

    if not selected_strategies:
        return [dbc.Alert("Please select at least one scheduling strategy.", color="warning")]

    if not all([num_devices, num_stages, num_batches, op_time_forward]):
         return [dbc.Alert("Missing required basic input values (Devices, Stages, Batches, Forward Time).", color="danger")]

    for strategy in selected_strategies:
        error_message = ""
        fig = go.Figure()
        placement_strategy = ""

        split_backward = strategy in ["zb1p", "dualpipe"]

        if split_backward and not all([op_time_backward_d, op_time_backward_w]):
            error_message = f"Strategy '{strategy}': Backward D and Backward W times are required."
        elif not split_backward and not op_time_backward:
            error_message = f"Strategy '{strategy}': Combined Backward time is required."

        if not error_message:
            if strategy in ["1f1b", "1f1b_overlap", "zb1p"]:
                placement_strategy = "standard"
                if num_devices != num_stages:
                    error_message = f"Strategy '{strategy}': Requires Number of Stages == Number of Devices."
            elif strategy in ["1f1b_interleave", "1f1b_interleave_overlap"]:
                placement_strategy = "interleave"
                if num_stages % num_devices != 0:
                    error_message = f"Strategy '{strategy}': Requires Number of Stages to be divisible by Number of Devices."
            elif strategy == "dualpipe":
                placement_strategy = "dualpipe"
                if num_stages % 2 != 0:
                    error_message = f"Strategy '{strategy}' (DualPipe): Requires an even number of stages."
                elif num_stages != num_devices:
                    error_message = f"Strategy '{strategy}' (DualPipe): Requires Number of Stages == Number of Devices."

        if not error_message:
            try:
                op_times = { "forward": float(op_time_forward) }
                if split_backward:
                    op_times["backward_D"] = float(op_time_backward_d)
                    op_times["backward_W"] = float(op_time_backward_w)
                    op_times["backward"] = float(op_time_backward_d) + float(op_time_backward_w)
                else:
                    op_times["backward"] = float(op_time_backward)

                config = ScheduleConfig(
                    num_devices=int(num_devices),
                    num_stages=int(num_stages),
                    num_batches=int(num_batches),
                    p2p_latency=float(p2p_latency),
                    placement_strategy=placement_strategy,
                    split_backward=split_backward,
                    op_times=op_times,
                )

                schedule_func = STRATEGIES.get(strategy)
                if not schedule_func:
                     raise ValueError(f"Invalid strategy function for: {strategy}")

                schedule = schedule_func(config)
                schedule.execute()

                vis_data = convert_schedule_to_visualization_format(schedule)
                fig = create_pipeline_figure(vis_data, show_progress=False)

                output_components.append(html.Div([
                    html.H4(f"Schedule: {strategy}", className="text-center mt-3 mb-2"),
                    dcc.Graph(figure=fig)
                ]))

            except (AssertionError, ValueError, TypeError) as e:
                 error_message = f"Error generating schedule for '{strategy}': {e}"
                 import traceback
                 traceback.print_exc()
            except Exception as e:
                 error_message = f"An unexpected error occurred for '{strategy}': {e}"
                 import traceback
                 traceback.print_exc()

        if error_message:
             output_components.append(
                 dbc.Alert(error_message, color="danger", className="mt-3")
             )

    return output_components

if __name__ == '__main__':
    port = 8050
    # Timer(1, open_browser, args=(port,)).start() # Optional: automatically open browser
    print(f"Dash server running on http://127.0.0.1:{port}")
    app.run_server(debug=True, port=port) 