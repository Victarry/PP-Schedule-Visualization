import dash
import dash_bootstrap_components as dbc
from dash import dcc, html, Input, Output, State, callback_context, ALL, ClientsideFunction
import plotly.graph_objects as go

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

STRATEGIES = {
    "1f1b": generate_1f1b_schedule,
    "zb1p": generate_zero_bubble_1p_schedule,
    "1f1b_overlap": generate_1f1b_overlap_schedule,
    "1f1b_interleave": generate_1f1b_interleave_schedule,
    "1f1b_interleave_overlap": generate_1f1b_interleave_overlap_schedule,
    "dualpipe": generate_dualpipe_schedule,
}

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP, dbc.icons.BOOTSTRAP], suppress_callback_exceptions=True)
app.title = "Pipeline Parallelism Schedule Visualizer"

# Initial default values
default_values = {
    "num_devices": 4,
    "num_stages": 8,
    "num_batches": 16,
    "p2p_latency": 0.0,
    "op_time_forward": 1.0,
    "op_time_backward_d": 1.0,
    "op_time_backward_w": 1.0,
    "op_time_backward": 2.0,
    "strategy": ["1f1b_interleave"],
    "op_time_overlapped_fwd_bwd": None,
}

# Define input groups using dbc components
card_style = {"marginBottom": "24px"}

basic_params_card = dbc.Card(
    dbc.CardBody([
        html.H5("Basic Parameters", className="card-title mb-4"),
        html.Div([
            dbc.Label("Number of Devices (GPUs)", html_for='num_devices', className="form-label"),
            dbc.Input(id='num_devices', type='number', value=default_values["num_devices"], min=1, step=1, required=True),
            dbc.FormFeedback("Please provide a positive integer for the number of devices.", type="invalid", id="feedback-num_devices"),
        ], className="mb-3"),
        html.Div([
            dbc.Label("Number of Stages (Model Chunks)", html_for='num_stages', className="form-label"),
            dbc.Input(id='num_stages', type='number', value=default_values["num_stages"], min=1, step=1, required=True),
            dbc.FormFeedback("Please provide a positive integer for the number of stages.", type="invalid", id="feedback-num_stages"),
        ], className="mb-3"),
        html.Div([
            dbc.Label("Number of Microbatches", html_for='num_batches', className="form-label"),
            dbc.Input(id='num_batches', type='number', value=default_values["num_batches"], min=1, step=1, required=True),
            dbc.FormFeedback("Please provide a positive integer for the number of microbatches.", type="invalid", id="feedback-num_batches"),
        ], className="mb-3"),
    ]),
    style=card_style
)

scheduling_params_card = dbc.Card(
    dbc.CardBody([
        html.H5("Scheduling Strategy", className="card-title mb-4"),
        dbc.ButtonGroup(
            [
                dbc.Button(
                    strategy,
                    id={"type": "strategy-button", "index": strategy},
                    color="secondary",
                    outline=True,
                    active=strategy in default_values["strategy"],
                    className="me-1"
                 )
                for strategy in STRATEGIES.keys()
            ],
            className="d-flex flex-wrap"
        ),
        dcc.Store(id='selected-strategies-store', data=default_values["strategy"]),
        html.Div(id='strategy-selection-feedback', className='invalid-feedback d-block mt-2')
    ]),
    style=card_style
)

timing_params_card = dbc.Card(
    dbc.CardBody([
        html.H5("Operation Timing (ms)", className="card-title mb-4"),
        html.Div([
            html.Div([
                dbc.Label("P2P Latency", html_for='p2p_latency', className="form-label d-inline-block me-1"),
                html.I(className="bi bi-info-circle", id="tooltip-target-p2p", style={"cursor": "pointer"})
            ]),
            dbc.Input(id='p2p_latency', type='number', value=default_values["p2p_latency"], min=0, step=0.01, required=True),
            dbc.FormFeedback("P2P latency must be a number >= 0.", type="invalid", id="feedback-p2p_latency"),
            dbc.Tooltip(
                "Time (ms) for point-to-point communication between adjacent devices.",
                target="tooltip-target-p2p",
                placement="right"
            )
        ], className="mb-3"),
        html.Div([
            html.Div([
                dbc.Label("Forward Operation Time", html_for='op_time_forward', className="form-label d-inline-block me-1"),
                html.I(className="bi bi-info-circle", id="tooltip-target-fwd", style={"cursor": "pointer"})
            ]),
            dbc.Input(id='op_time_forward', type='number', value=default_values["op_time_forward"], min=0.01, step=0.01, required=True),
            dbc.FormFeedback("Forward time must be a number > 0.", type="invalid", id="feedback-op_time_forward"),
            dbc.Tooltip(
                "Time (ms) for a single forward pass of one microbatch through one stage.",
                target="tooltip-target-fwd",
                placement="right"
            )
        ], className="mb-3"),
        html.Div([
            html.Div([
                dbc.Label("Backward (Combined)", html_for='op_time_backward', className="form-label d-inline-block me-1"),
                html.I(className="bi bi-info-circle", id="tooltip-target-bwd", style={"cursor": "pointer"})
            ]),
            dbc.Input(id='op_time_backward', type='number', value=default_values["op_time_backward"], min=0.01, step=0.01),
            dbc.FormText("Used when strategy does NOT require split backward."),
            dbc.FormFeedback("Backward time must be > 0 if specified.", type="invalid", id="feedback-op_time_backward"),
            dbc.Tooltip(
                "Time (ms) for a combined backward pass (data gradient + weight gradient) of one microbatch through one stage.",
                target="tooltip-target-bwd",
                placement="right"
            )
        ], className="mb-3"),

        # --- Collapsible Advanced Options (Item 3) ---
        html.Hr(className="my-3"),
        dbc.Switch(
            id="advanced-timing-switch",
            label="Show Advanced Timing Options",
            value=False,
            className="mb-3"
        ),
        dbc.Collapse(
            id="advanced-timing-collapse",
            is_open=False,
            children=[
                html.Div([
                    html.Div([
                        dbc.Label("Backward D (Data Grad)", html_for='op_time_backward_d', className="form-label d-inline-block me-1"),
                        html.I(className="bi bi-info-circle", id="tooltip-target-bwd-d", style={"cursor": "pointer"})
                    ]),
                    dbc.Input(id='op_time_backward_d', type='number', value=default_values["op_time_backward_d"], min=0.01, step=0.01),
                    dbc.FormText("Used when strategy requires split backward (e.g., ZB-1P, DualPipe)."),
                    dbc.FormFeedback("Backward D time must be > 0 if specified.", type="invalid", id="feedback-op_time_backward_d"),
                    dbc.Tooltip(
                        "Time (ms) for the data gradient part of the backward pass.",
                        target="tooltip-target-bwd-d",
                        placement="right"
                    )
                ], className="mb-3"),
                html.Div([
                    html.Div([
                        dbc.Label("Backward W (Weight Grad)", html_for='op_time_backward_w', className="form-label d-inline-block me-1"),
                        html.I(className="bi bi-info-circle", id="tooltip-target-bwd-w", style={"cursor": "pointer"})
                    ]),
                    dbc.Input(id='op_time_backward_w', type='number', value=default_values["op_time_backward_w"], min=0.01, step=0.01),
                    dbc.FormText("Used when strategy requires split backward (e.g., ZB-1P, DualPipe)."),
                    dbc.FormFeedback("Backward W time must be > 0 if specified.", type="invalid", id="feedback-op_time_backward_w"),
                    dbc.Tooltip(
                        "Time (ms) for the weight gradient part of the backward pass.",
                        target="tooltip-target-bwd-w",
                        placement="right"
                    )
                ], className="mb-3"),
                html.Div([
                    html.Div([
                        dbc.Label("Overlapped Forward+Backward", html_for='op_time_overlapped_fwd_bwd', className="form-label d-inline-block me-1"),
                        html.I(className="bi bi-info-circle", id="tooltip-target-overlap", style={"cursor": "pointer"})
                    ]),
                    dbc.Input(id='op_time_overlapped_fwd_bwd', type='number', placeholder="Defaults to Fwd + Bwd", min=0.01, step=0.01, value=default_values["op_time_overlapped_fwd_bwd"]),
                    dbc.FormText("Specify if Forward and Backward ops overlap completely."),
                    dbc.FormFeedback("Overlapped time must be > 0 if specified.", type="invalid", id="feedback-op_time_overlapped_fwd_bwd"),
                    dbc.Tooltip(
                        "Optional: Specify a single time (ms) if the forward and backward passes for a microbatch can be fully overlapped within the same stage execution slot.",
                        target="tooltip-target-overlap",
                        placement="right"
                    )
                ], className="mb-3"),
            ]
        )
    ]),
    style=card_style
)

# Updated app layout using dbc components and structure
app.layout = dbc.Container([
    html.H1("Pipeline Parallelism Schedule Visualizer", className="my-4 text-center"),

    # Main Row with Left (Graphs) and Right (Controls) Columns
    dbc.Row([
        # --- Left Column (Graphs Area) ---
        dbc.Col([
            # Output Area for Graphs
            dcc.Loading(
                id="loading-graph-area",
                type="circle",
                children=html.Div(id='graph-output-container', style={"minHeight": "600px"})
            )
        ], lg=8, md=7, sm=12, className="mb-4 mb-lg-0"),

        # --- Right Column (Controls Area) ---
        dbc.Col([
            # Parameter Cards Stacked Vertically
            basic_params_card,
            scheduling_params_card,
            timing_params_card,

            # Generate Button below the cards in the right column
            dbc.Row([
                dbc.Col(
                    dbc.Button(
                        'Generate Schedule',
                        id='generate-button',
                        n_clicks=0,
                        color="primary",
                        className="w-100",
                        disabled=False
                    ),
                )
            ], className="mt-3")
        ], lg=4, md=5, sm=12)
    ]),

    # --- Toast Container (Positioned Fixed) ---
    html.Div(id="toast-container", style={"position": "fixed", "top": 20, "right": 20, "zIndex": 1050})

], fluid=True, className="py-4")

# --- Callback for Input Validation and Generate Button State ---
@app.callback(
    Output('generate-button', 'disabled'),
    # Outputs to control the 'invalid' state of Inputs
    Output('num_devices', 'invalid'),
    Output('num_stages', 'invalid'),
    Output('num_batches', 'invalid'),
    Output('p2p_latency', 'invalid'),
    Output('op_time_forward', 'invalid'),
    Output('op_time_backward', 'invalid'),
    Output('op_time_backward_d', 'invalid'),
    Output('op_time_backward_w', 'invalid'),
    Output('op_time_overlapped_fwd_bwd', 'invalid'),
    # Outputs to control the visibility/content of FormFeedback (can also just control Input's invalid state)
    # We are primarily using the Input's `invalid` prop which automatically shows/hides associated FormFeedback
    # Output('feedback-num_devices', 'children'), ... (Add if more specific messages needed per validation type)
    Output('strategy-selection-feedback', 'children', allow_duplicate=True), # Update feedback from validation callback too
    # Inputs: Trigger validation whenever any relevant input changes
    Input('num_devices', 'value'),
    Input('num_stages', 'value'),
    Input('num_batches', 'value'),
    Input('p2p_latency', 'value'),
    Input('op_time_forward', 'value'),
    Input('op_time_backward', 'value'),
    Input('op_time_backward_d', 'value'),
    Input('op_time_backward_w', 'value'),
    Input('op_time_overlapped_fwd_bwd', 'value'),
    Input('selected-strategies-store', 'data'), # Validate strategy selection
    prevent_initial_call=True # Prevent callback running on page load before user interaction
)
def validate_inputs(num_devices, num_stages, num_batches, p2p_latency,
                    op_time_forward, op_time_backward, op_time_backward_d, op_time_backward_w,
                    op_time_overlapped_fwd_bwd, selected_strategies):
    is_invalid = {
        "num_devices": num_devices is None or num_devices < 1,
        "num_stages": num_stages is None or num_stages < 1,
        "num_batches": num_batches is None or num_batches < 1,
        "p2p_latency": p2p_latency is None or p2p_latency < 0,
        "op_time_forward": op_time_forward is None or op_time_forward <= 0,
        "op_time_backward": op_time_backward is not None and op_time_backward <= 0,
        "op_time_backward_d": op_time_backward_d is not None and op_time_backward_d <= 0,
        "op_time_backward_w": op_time_backward_w is not None and op_time_backward_w <= 0,
        "op_time_overlapped_fwd_bwd": op_time_overlapped_fwd_bwd is not None and op_time_overlapped_fwd_bwd <= 0,
    }

    # Validate strategy selection
    strategy_feedback = "" # Default empty feedback
    if not selected_strategies or len(selected_strategies) == 0:
        is_invalid["strategies"] = True
        strategy_feedback = "Please select at least one strategy."
    else:
        is_invalid["strategies"] = False
        # Additional validation: Check if required timings are provided for selected strategies
        needs_split_backward = any(s in ["zb1p", "dualpipe"] for s in selected_strategies)
        needs_combined_backward = any(s not in ["zb1p", "dualpipe"] for s in selected_strategies)

        if needs_split_backward and (op_time_backward_d is None or op_time_backward_w is None):
            is_invalid["op_time_backward_d"] = op_time_backward_d is None or op_time_backward_d <= 0
            is_invalid["op_time_backward_w"] = op_time_backward_w is None or op_time_backward_w <= 0
            # We might want specific feedback here, but setting invalid=True is often enough

        if needs_combined_backward and op_time_backward is None:
            is_invalid["op_time_backward"] = op_time_backward is None or op_time_backward <= 0

    # Check if any input is invalid
    overall_invalid = any(is_invalid.values())

    # Disable button if any validation fails
    disable_button = overall_invalid

    # Return button state and invalid states for each input
    return (
        disable_button,
        is_invalid["num_devices"],
        is_invalid["num_stages"],
        is_invalid["num_batches"],
        is_invalid["p2p_latency"],
        is_invalid["op_time_forward"],
        is_invalid["op_time_backward"],
        is_invalid["op_time_backward_d"],
        is_invalid["op_time_backward_w"],
        is_invalid["op_time_overlapped_fwd_bwd"],
        strategy_feedback # Update strategy feedback based on validation
    )

# --- Callback to toggle Advanced Options Collapse ---
@app.callback(
    Output("advanced-timing-collapse", "is_open"),
    Input("advanced-timing-switch", "value"),
    prevent_initial_call=True,
)
def toggle_advanced_options(switch_value):
    return switch_value

# --- Client-side Callback for Strategy ButtonGroup ---
app.clientside_callback(
    ClientsideFunction(
        namespace='clientside',
        function_name='update_strategy_selection'
    ),
    Output('selected-strategies-store', 'data'),
    Output({'type': 'strategy-button', 'index': ALL}, 'active'),
    Output({'type': 'strategy-button', 'index': ALL}, 'color'),
    Output({'type': 'strategy-button', 'index': ALL}, 'outline'),
    Output('strategy-selection-feedback', 'children'),
    Input({'type': 'strategy-button', 'index': ALL}, 'n_clicks'),
    State('selected-strategies-store', 'data'),
    prevent_initial_call=True
)

# --- Main Graph Update Callback ---
@app.callback(
    # Output graph container and toast container separately
    Output('graph-output-container', 'children'),
    Output('toast-container', 'children'), # Output for toasts
    Input('generate-button', 'n_clicks'),
    State('num_devices', 'value'),
    State('num_stages', 'value'),
    State('num_batches', 'value'),
    State('p2p_latency', 'value'),
    State('op_time_forward', 'value'),
    State('op_time_backward', 'value'),
    State('op_time_backward_d', 'value'),
    State('op_time_backward_w', 'value'),
    State('op_time_overlapped_fwd_bwd', 'value'),
    State('selected-strategies-store', 'data'),
    prevent_initial_call=True
)
def update_graph(n_clicks, num_devices, num_stages, num_batches, p2p_latency,
                 op_time_forward, op_time_backward, op_time_backward_d, op_time_backward_w,
                 op_time_overlapped_fwd_bwd,
                 selected_strategies):

    strategy_display_order = ["1f1b", "1f1b_interleave", "1f1b_overlap", "1f1b_interleave_overlap", "dualpipe", "zb1p"]

    graph_components = []
    toast_components = []
    valid_results = []
    error_messages = []
    automatic_adjustments = []
    execution_times = [] # Add list to store execution times

    # Use a variable to track if initial validation fails
    initial_validation_error = None

    if not selected_strategies:
        initial_validation_error = dbc.Toast(
            "Please select at least one scheduling strategy.",
            header="Input Error",
            icon="warning",
            duration=4000,
            is_open=True,
            className="border-warning"
        )
    elif not all([num_devices, num_stages, num_batches, op_time_forward]):
        initial_validation_error = dbc.Toast(
            "Missing required basic input values (Devices, Stages, Batches, Forward Time).",
            header="Input Error",
            icon="danger",
            duration=4000,
            is_open=True,
            className="border-danger"
        )

    if initial_validation_error:
        # Return empty graph list and the validation error toast
        return [], [initial_validation_error]

    for strategy in selected_strategies:
        error_message = ""
        placement_strategy = ""
        
        # Use local copies of params that might be adjusted for this strategy
        current_num_stages = num_stages
        current_num_devices = num_devices
        
        # Apply automatic adjustments for dualpipe
        if strategy == "dualpipe" and num_stages != num_devices:
            current_num_stages = num_devices
            adjustment_msg = f"Strategy '{strategy}': Number of Stages auto-adjusted to {num_devices} to match Devices."
            automatic_adjustments.append(adjustment_msg)

        # Apply automatic adjustments for strategies that require num_stages == num_devices
        if strategy in ["1f1b", "1f1b_overlap", "zb1p"] and num_stages != num_devices:
            current_num_stages = num_devices
            adjustment_msg = f"Strategy '{strategy}': Number of Stages auto-adjusted to {num_devices} to match Devices."
            automatic_adjustments.append(adjustment_msg)

        split_backward = strategy in ["zb1p", "dualpipe"]

        if split_backward and not all([op_time_backward_d, op_time_backward_w]):
            error_message = f"Strategy '{strategy}': Backward D and Backward W times are required."
        elif not split_backward and not op_time_backward:
            error_message = f"Strategy '{strategy}': Combined Backward time is required."

        if not error_message:
            if strategy in ["1f1b", "1f1b_overlap", "zb1p"]:
                placement_strategy = "standard"
            elif strategy in ["1f1b_interleave", "1f1b_interleave_overlap"]:
                placement_strategy = "interleave"
                if current_num_stages % current_num_devices != 0:
                    error_message = f"Strategy '{strategy}': Requires Stages divisible by Devices."
            elif strategy == "dualpipe":
                placement_strategy = "dualpipe"
                if current_num_stages % 2 != 0:
                    error_message = f"Strategy '{strategy}': Requires an even number of stages."

        # Create adjusted operation times based on placement strategy
        if not error_message:
            try:
                stages_per_device = current_num_stages // current_num_devices
                time_scale_factor = 1.0 / stages_per_device if stages_per_device > 0 else 1.0

                if stages_per_device > 1:
                     adjustment_msg = f"Strategy '{strategy}': Op times scaled by 1/{stages_per_device} ({stages_per_device} stages/device)."
                     # Avoid adding duplicate adjustment messages if already added above
                     if adjustment_msg not in automatic_adjustments:
                         automatic_adjustments.append(adjustment_msg)

                op_times = { "forward": float(op_time_forward) * time_scale_factor }

                if split_backward:
                    op_times["backward_D"] = float(op_time_backward_d) * time_scale_factor
                    op_times["backward_W"] = float(op_time_backward_w) * time_scale_factor
                    op_times["backward"] = (float(op_time_backward_d) + float(op_time_backward_w)) * time_scale_factor
                else:
                    op_times["backward"] = float(op_time_backward) * time_scale_factor

                if op_time_overlapped_fwd_bwd is not None:
                    try:
                        overlapped_val = float(op_time_overlapped_fwd_bwd)
                        if overlapped_val > 0:
                             op_times["overlapped_forward_backward"] = overlapped_val * time_scale_factor
                    except (ValueError, TypeError):
                         pass

                config = ScheduleConfig(
                    num_devices=int(current_num_devices),
                    num_stages=int(current_num_stages),
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
                valid_results.append((strategy, schedule, vis_data))
                # Store execution time
                execution_times.append((strategy, schedule.get_total_execution_time()))

            except (AssertionError, ValueError, TypeError) as e:
                 error_message = f"Error for '{strategy}': {e}"
            except Exception as e:
                 error_message = f"Unexpected error for '{strategy}': {e}"

        if error_message:
             error_messages.append((strategy, error_message))

    # --- Generate Toasts --- 
    # Add toasts for automatic adjustments
    for adjustment in automatic_adjustments:
        toast_components.append(
            dbc.Toast(
                adjustment,
                header="Parameter Adjustment",
                icon="info",
                duration=5000, # Slightly longer duration for info
                is_open=True,
                className="border-info"
            )
        )

    # Add toasts for errors
    for strategy, msg in error_messages:
        toast_components.append(
            dbc.Toast(
                msg,
                header=f"Error: {strategy}",
                icon="danger",
                duration=8000, # Longer duration for errors
                is_open=True,
                className="border-danger"
            )
        )
    
    # --- Generate Graphs ---
    if valid_results:
        max_execution_time = max(schedule.get_total_execution_time() for _, schedule, _ in valid_results)
        sorted_valid_results = sorted(valid_results, key=lambda x: strategy_display_order.index(x[0]) if x[0] in strategy_display_order else float('inf'))

        # Prepare graphs for single-column layout
        graph_components = [] # Use graph_components again
        for strategy, _, vis_data in sorted_valid_results:
            fig = create_pipeline_figure(vis_data, max_time=max_execution_time, show_progress=False)
            margin = max_execution_time * 0.05
            fig.update_layout(
                xaxis=dict(range=[0, max_execution_time + margin])
            )
            # Append the Div directly for vertical stacking
            graph_components.append(
                html.Div([
                    html.H4(f"Schedule: {strategy}", className="text-center mt-3 mb-2"),
                    dcc.Graph(figure=fig)
                ])
            )

        # No grid arrangement needed for single column
        # rows = [] ... removed ...

    # If there are graphs, use the component list, otherwise show a message
    output_content = []
    if graph_components: # Check if graph_components list is populated
        output_content = graph_components # Assign the list of components
    elif not toast_components: # Only show 'no results' if no errors/adjustments either
        output_content = dbc.Alert("Click 'Generate Schedule' to see results.", color="info", className="mt-3")

    # Add the execution time table if there are results
    if execution_times:
        # Sort times based on execution time (ascending)
        sorted_times = sorted(execution_times, key=lambda x: x[1])
        min_time = sorted_times[0][1] if sorted_times else None

        table_header = [html.Thead(html.Tr([html.Th("Strategy"), html.Th("Total Execution Time (ms)")]))]
        table_rows = []
        for strategy, time in sorted_times:
            row_class = "table-success" if time == min_time else ""
            table_rows.append(html.Tr([html.Td(strategy), html.Td(f"{time:.2f}")], className=row_class))

        table_body = [html.Tbody(table_rows)]
        summary_table = dbc.Table(
            table_header + table_body,
            bordered=True,
            striped=True,
            hover=True,
            responsive=True,
            color="light", # Apply a light theme color
            className="mt-5" # Add margin top
        )
        # Prepend title to the table
        table_component = html.Div([
            html.H4("Execution Time Summary", className="text-center mt-4 mb-3"),
            summary_table
        ])

        # Append the table component to the output content
        # If output_content is just the alert, replace it. Otherwise, append.
        if isinstance(output_content, list):
            output_content.append(table_component)
        else: # It must be the Alert
            output_content = [output_content, table_component] # Replace Alert with list

    # Return graph components (single column list or message) and toast components
    return output_content, toast_components

# For Hugging Face Spaces deployment
server = app.server

if __name__ == '__main__':
    app.run_server(debug=False, host='0.0.0.0', port=7860) 