import dash
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

app = dash.Dash(__name__, suppress_callback_exceptions=True)
app.title = "Pipeline Parallelism Visualizer"

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

app.layout = html.Div([
    html.H1("Pipeline Parallelism Schedule Visualizer", style={'textAlign': 'center'}),

    html.Div([
        html.Div([
            html.Label("Number of Devices (GPUs):"),
            dcc.Input(id='num_devices', type='number', value=default_values["num_devices"], min=1, step=1, style={'width': '100%'}),

            html.Label("Number of Stages (Model Chunks):"),
            dcc.Input(id='num_stages', type='number', value=default_values["num_stages"], min=1, step=1, style={'width': '100%'}),

            html.Label("Number of Microbatches:"),
            dcc.Input(id='num_batches', type='number', value=default_values["num_batches"], min=1, step=1, style={'width': '100%'}),

            html.Label("P2P Latency (ms):"),
            dcc.Input(id='p2p_latency', type='number', value=default_values["p2p_latency"], min=0, step=0.01, style={'width': '100%'}),

        ], style={'padding': 10, 'flex': 1}),

        html.Div([
            html.Label("Scheduling Strategy:"),
            dcc.Dropdown(
                id='strategy',
                options=[{'label': k, 'value': k} for k in STRATEGIES.keys()],
                value=default_values["strategy"],
                clearable=False,
                style={'width': '100%'}
            ),

            html.Label("Placement Strategy:"),
            dcc.Dropdown(
                id='placement_strategy',
                options=[
                    {'label': 'Standard', 'value': 'standard'},
                    {'label': 'Interleave', 'value': 'interleave'},
                    {'label': 'DualPipe', 'value': 'dualpipe'}
                ],
                value=default_values["placement_strategy"],
                clearable=False,
                style={'width': '100%'}
            ),

            html.Div([ # Wrap checkbox and label
                dcc.Checklist(
                    id='split_backward',
                    options=[{'label': ' Split Backward Pass (for ZB-1P, DualPipe)', 'value': 'True'}],
                    value=['True'] if default_values["split_backward"] else [],
                    style={'display': 'inline-block'}
                ),
            ], style={'marginTop': '20px'}),

        ], style={'padding': 10, 'flex': 1}),

        html.Div([
            html.Label("Operation Time - Forward (ms):"),
            dcc.Input(id='op_time_forward', type='number', value=default_values["op_time_forward"], min=0.01, step=0.01, style={'width': '100%'}),

            html.Label("Operation Time - Backward (ms):"),
            dcc.Input(id='op_time_backward', type='number', value=default_values["op_time_backward"], min=0.01, step=0.01, style={'width': '100%'}),

            html.Label("Operation Time - Backward D (Data Grad) (ms):"),
            dcc.Input(id='op_time_backward_d', type='number', value=default_values["op_time_backward_d"], min=0.01, step=0.01, style={'width': '100%'}),

            html.Label("Operation Time - Backward W (Weight Grad) (ms):"),
            dcc.Input(id='op_time_backward_w', type='number', value=default_values["op_time_backward_w"], min=0.01, step=0.01, style={'width': '100%'}),
        ], style={'padding': 10, 'flex': 1}),

    ], style={'display': 'flex', 'flexDirection': 'row'}),

    html.Div([
        html.Button('Generate Schedule', id='generate-button', n_clicks=0, style={'margin': '20px auto', 'display': 'block'}),
    ]),

    html.Div(id='error-message', style={'color': 'red', 'textAlign': 'center', 'marginTop': '10px'}),

    dcc.Loading(
        id="loading-graph",
        type="circle",
        children=dcc.Graph(id='pipeline-graph', figure=go.Figure())
    )
])

@app.callback(
    Output('pipeline-graph', 'figure'),
    Output('error-message', 'children'),
    Input('generate-button', 'n_clicks'),
    State('num_devices', 'value'),
    State('num_stages', 'value'),
    State('num_batches', 'value'),
    State('p2p_latency', 'value'),
    State('op_time_forward', 'value'),
    State('op_time_backward', 'value'),
    State('op_time_backward_d', 'value'),
    State('op_time_backward_w', 'value'),
    State('strategy', 'value'),
    State('split_backward', 'value'),
    State('placement_strategy', 'value'),
    prevent_initial_call=True
)
def update_graph(n_clicks, num_devices, num_stages, num_batches, p2p_latency,
                 op_time_forward, op_time_backward, op_time_backward_d, op_time_backward_w,
                 strategy, split_backward_list, placement_strategy):

    error_message = ""
    fig = go.Figure()

    split_backward = 'True' in split_backward_list

    # Basic Validations
    if not all([num_devices, num_stages, num_batches, op_time_forward]):
        return fig, "Missing required input values."
    if split_backward and not all([op_time_backward_d, op_time_backward_w]):
        return fig, "Backward D and Backward W times are required when 'Split Backward' is checked."
    if not split_backward and not op_time_backward:
        return fig, "Backward time is required when 'Split Backward' is unchecked."
    if num_stages % num_devices != 0 and placement_strategy != 'dualpipe':
         return fig, "Number of Stages must be divisible by Number of Devices for standard/interleave placement."
    if placement_strategy == 'dualpipe' and num_stages % 2 != 0:
        return fig, "DualPipe requires an even number of stages."
    if placement_strategy == 'dualpipe' and num_stages != num_devices:
        return fig, "DualPipe requires Number of Stages to be equal to Number of Devices."
    if strategy == 'dualpipe' and not split_backward:
        return fig, "DualPipe strategy currently requires 'Split Backward' to be checked."
    if strategy == 'dualpipe' and placement_strategy != 'dualpipe':
        return fig, "DualPipe strategy requires 'DualPipe' placement strategy."
    if strategy == 'zb1p' and not split_backward:
        return fig, "ZB-1P strategy requires 'Split Backward' to be checked."

    try:
        op_times = {
            "forward": float(op_time_forward),
        }
        if split_backward:
            op_times["backward_D"] = float(op_time_backward_d)
            op_times["backward_W"] = float(op_time_backward_w)
            # Add combined backward time for compatibility if needed by some visualization or calculation
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
            raise ValueError(f"Invalid strategy selected: {strategy}")

        schedule = schedule_func(config)
        schedule.execute() # Calculate start/end times

        vis_data = convert_schedule_to_visualization_format(schedule)
        fig = create_pipeline_figure(vis_data, show_progress=False) # Disable progress bar in server mode

    except AssertionError as e:
        error_message = f"Configuration Error: {e}"
        fig = go.Figure() # Return empty figure on error
    except ValueError as e:
        error_message = f"Input Error: {e}"
        fig = go.Figure()
    except Exception as e:
        error_message = f"An unexpected error occurred: {e}"
        fig = go.Figure()

    return fig, error_message

if __name__ == '__main__':
    port = 8050
    # Timer(1, open_browser, args=(port,)).start() # Optional: automatically open browser
    print(f"Dash server running on http://127.0.0.1:{port}")
    app.run_server(debug=True, port=port) 