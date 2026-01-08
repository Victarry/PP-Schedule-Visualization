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

# Strategy descriptions for better UX
STRATEGY_INFO = {
    "1f1b": {
        "name": "1F1B",
        "description": "One Forward One Backward - Standard pipeline parallelism",
        "icon": "bi-arrow-left-right"
    },
    "zb1p": {
        "name": "ZB-1P",
        "description": "Zero Bubble 1-stage Pipeline - Minimizes pipeline bubbles",
        "icon": "bi-circle"
    },
    "1f1b_overlap": {
        "name": "1F1B Overlap",
        "description": "1F1B with overlapped forward and backward passes",
        "icon": "bi-layers"
    },
    "1f1b_interleave": {
        "name": "1F1B Interleave",
        "description": "Interleaved pipeline stages for better efficiency",
        "icon": "bi-shuffle"
    },
    "1f1b_interleave_overlap": {
        "name": "1F1B Interleave + Overlap",
        "description": "Combines interleaving with overlapped execution",
        "icon": "bi-layers-fill"
    },
    "dualpipe": {
        "name": "DualPipe",
        "description": "Dual pipeline execution for enhanced parallelism",
        "icon": "bi-diagram-2"
    }
}

app = dash.Dash(
    __name__, 
    external_stylesheets=[
        dbc.themes.BOOTSTRAP, 
        dbc.icons.BOOTSTRAP,
        "https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap"
    ], 
    suppress_callback_exceptions=True
)
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
    "microbatch_group_size_per_vp_stage": None,
}

# Define input groups using dbc components
card_style = {"marginBottom": "20px"}

# Header section
header = html.Div([
    html.Div([
        html.H1([
            html.I(className="bi bi-diagram-3 me-3"),
            "Pipeline Parallelism Schedule Visualizer"
        ], className="text-center mb-0"),
        html.P("Visualize and compare different pipeline parallelism scheduling strategies", 
               className="text-center mt-2 mb-0 lead")
    ], className="container")
], className="main-header")

# Basic parameters card with improved styling
basic_params_card = dbc.Card([
    dbc.CardBody([
        html.H5([
            html.I(className="bi bi-sliders section-icon"),
            "Basic Configuration"
        ], className="section-title"),
        
        dbc.Row([
            dbc.Col([
                dbc.Label("Number of Devices (GPUs)", html_for='num_devices', className="form-label"),
                dbc.InputGroup([
                    dbc.InputGroupText(html.I(className="bi bi-gpu-card")),
                    dbc.Input(id='num_devices', type='number', value=default_values["num_devices"], 
                             min=1, step=1, required=True),
                ]),
                dbc.FormFeedback("Please provide a positive integer.", type="invalid"),
            ], md=6),
            
            dbc.Col([
                dbc.Label("Number of Stages", html_for='num_stages', className="form-label"),
                dbc.InputGroup([
                    dbc.InputGroupText(html.I(className="bi bi-stack")),
                    dbc.Input(id='num_stages', type='number', value=default_values["num_stages"], 
                             min=1, step=1, required=True),
                ]),
                dbc.FormFeedback("Please provide a positive integer.", type="invalid"),
            ], md=6),
        ], className="mb-3"),
        
        dbc.Row([
            dbc.Col([
                dbc.Label("Number of Microbatches", html_for='num_batches', className="form-label"),
                dbc.InputGroup([
                    dbc.InputGroupText(html.I(className="bi bi-collection")),
                    dbc.Input(id='num_batches', type='number', value=default_values["num_batches"], 
                             min=1, step=1, required=True),
                ]),
                dbc.FormFeedback("Please provide a positive integer.", type="invalid"),
            ], md=12),
        ]),
    ])
], style=card_style)

# Improved scheduling strategy selection
scheduling_params_card = dbc.Card([
    dbc.CardBody([
        html.H5([
            html.I(className="bi bi-diagram-2 section-icon"),
            "Scheduling Strategy"
        ], className="section-title"),
        
        html.P("Select one or more strategies to compare:", className="text-muted mb-3"),
        
        dbc.Row([
            dbc.Col([
                html.Div([
                    html.Div([
                        html.I(className=f"{STRATEGY_INFO[strategy]['icon']} mb-2"),
                        html.H6(STRATEGY_INFO[strategy]['name'], className="mb-1"),
                        html.Small(STRATEGY_INFO[strategy]['description'], className="text-muted")
                    ], 
                    id={"type": "strategy-card", "index": strategy},
                    className=f"strategy-card p-3 text-center {'selected' if strategy in default_values['strategy'] else ''}",
                    )
                ], className="mb-3")
            ], lg=4, md=6) for strategy in STRATEGIES.keys()
        ], className="g-3"),
        
        dcc.Store(id='selected-strategies-store', data=default_values["strategy"]),
        html.Div(id='strategy-selection-feedback', className='invalid-feedback d-block mt-2')
    ])
], style=card_style)

# Timing parameters with better organization
timing_params_card = dbc.Card([
    dbc.CardBody([
        html.H5([
            html.I(className="bi bi-clock section-icon"),
            "Operation Timing Configuration"
        ], className="section-title"),
        
        # Basic timing parameters
        dbc.Row([
            dbc.Col([
                html.Div([
                    dbc.Label([
                        "P2P Latency (ms)",
                        dbc.Badge("?", pill=True, color="secondary", className="ms-1", id="tooltip-p2p",
                                 style={"cursor": "pointer", "fontSize": "0.75rem"})
                    ], html_for='p2p_latency', className="form-label"),
                    dbc.Input(id='p2p_latency', type='number', value=default_values["p2p_latency"], 
                             min=0, step=0.01, required=True),
                    dbc.FormFeedback("Must be ≥ 0", type="invalid"),
                    dbc.Tooltip(
                        "Time for point-to-point communication between adjacent devices.",
                        target="tooltip-p2p",
                        placement="right"
                    )
                ])
            ], md=6),
            
            dbc.Col([
                html.Div([
                    dbc.Label([
                        "Forward Pass Time (ms)",
                        dbc.Badge("?", pill=True, color="secondary", className="ms-1", id="tooltip-fwd",
                                 style={"cursor": "pointer", "fontSize": "0.75rem"})
                    ], html_for='op_time_forward', className="form-label"),
                    dbc.Input(id='op_time_forward', type='number', value=default_values["op_time_forward"], 
                             min=0.01, step=0.01, required=True),
                    dbc.FormFeedback("Must be > 0", type="invalid"),
                    dbc.Tooltip(
                        "Time for a forward pass of one microbatch through one stage.",
                        target="tooltip-fwd",
                        placement="right"
                    )
                ])
            ], md=6),
        ], className="mb-3"),
        
        # Backward timing
        html.Div([
            dbc.Label([
                "Backward Pass Time (ms)",
                dbc.Badge("?", pill=True, color="secondary", className="ms-1", id="tooltip-bwd",
                         style={"cursor": "pointer", "fontSize": "0.75rem"})
            ], html_for='op_time_backward', className="form-label"),
            dbc.Input(id='op_time_backward', type='number', value=default_values["op_time_backward"], 
                     min=0.01, step=0.01),
            dbc.FormText("Combined backward pass time (data + weight gradients)", className="mt-1"),
            dbc.FormFeedback("Must be > 0", type="invalid"),
            dbc.Tooltip(
                "Time for combined backward pass (data + weight gradients).",
                target="tooltip-bwd",
                placement="right"
            )
        ], className="mb-3"),

        # Advanced options with better styling
        html.Hr(className="my-4"),
        dbc.Button([
            html.I(className="bi bi-gear-fill me-2"),
            "Advanced Timing Options"
        ], 
        id="advanced-timing-toggle",
        color="light",
        className="mb-3",
        size="sm"
        ),
        
        dbc.Collapse([
            dbc.Alert([
                html.I(className="bi bi-info-circle-fill me-2"),
                "These options are for advanced users and specific scheduling strategies."
            ], color="info", className="mb-3"),
            
            dbc.Row([
                dbc.Col([
                    dbc.Label("Backward D (Data Gradient)", html_for='op_time_backward_d'),
                    dbc.Input(id='op_time_backward_d', type='number', 
                             value=default_values["op_time_backward_d"], min=0.01, step=0.01),
                    dbc.FormText("For strategies with split backward"),
                    dbc.FormFeedback("Must be > 0", type="invalid"),
                ], md=6),
                
                dbc.Col([
                    dbc.Label("Backward W (Weight Gradient)", html_for='op_time_backward_w'),
                    dbc.Input(id='op_time_backward_w', type='number', 
                             value=default_values["op_time_backward_w"], min=0.01, step=0.01),
                    dbc.FormText("For strategies with split backward"),
                    dbc.FormFeedback("Must be > 0", type="invalid"),
                ], md=6),
            ], className="mb-3"),
            
            html.Div([
                dbc.Label("Overlapped Forward+Backward Time", html_for='op_time_overlapped_fwd_bwd'),
                dbc.Input(id='op_time_overlapped_fwd_bwd', type='number', 
                         placeholder="Auto-calculated if not specified", min=0.01, step=0.01),
                dbc.FormText("Time when forward and backward can be fully overlapped"),
                dbc.FormFeedback("Must be > 0", type="invalid"),
            ], className="mb-3"),
            
            html.Div([
                dbc.Label("Microbatch Group Size per VP Stage", html_for='microbatch_group_size_per_vp_stage'),
                dbc.Input(id='microbatch_group_size_per_vp_stage', type='number', 
                         placeholder=f"Defaults to number of devices", min=1, step=1),
                dbc.FormText("For interleave strategies only"),
                dbc.FormFeedback("Must be a positive integer", type="invalid"),
            ]),
        ], id="advanced-timing-collapse", is_open=False)
    ])
], style=card_style)

# Per-stage timing configuration card
per_stage_timing_card = dbc.Card([
    dbc.CardBody([
        html.H5([
            html.I(className="bi bi-list-ol section-icon"),
            "Per-Stage Timing Configuration"
        ], className="section-title"),
        
        dbc.Button([
            html.I(className="bi bi-sliders2 me-2"),
            "Customize Per-Stage Timing"
        ], 
        id="per-stage-timing-toggle",
        color="light",
        className="mb-3 w-100",
        size="sm"
        ),
        
        dbc.Collapse([
            dbc.Alert([
                html.I(className="bi bi-info-circle-fill me-2"),
                "Override global timing values for individual stages. Leave empty to use global values."
            ], color="info", className="mb-3"),
            
            html.Div(id='per-stage-inputs-container', children=[])
        ], id="per-stage-timing-collapse", is_open=False)
    ])
], style=card_style)

# Updated app layout with improved structure
app.layout = html.Div([
    header,
    
    dbc.Container([
        dbc.Row([
            # Left Column - Visualization Area
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H5([
                            html.I(className="bi bi-graph-up section-icon"),
                            "Visualization Results"
                        ], className="section-title"),
                        
                        dcc.Loading(
                            id="loading-graph-area",
                            type="circle",
                            children=html.Div([
                                # Welcome message for initial state
                                dbc.Alert([
                                    html.H4([
                                        html.I(className="bi bi-lightbulb me-2"),
                                        "Welcome to Pipeline Parallelism Schedule Visualizer"
                                    ], className="alert-heading"),
                                    html.Hr(),
                                    html.P([
                                        "This tool helps you visualize and compare different pipeline parallelism scheduling strategies. ",
                                        "To get started:"
                                    ], className="mb-3"),
                                    html.Ol([
                                        html.Li("Configure your basic parameters (devices, stages, microbatches)"),
                                        html.Li("Select one or more scheduling strategies to compare"),
                                        html.Li("Set the operation timing parameters"),
                                        html.Li("Click 'Generate Schedules' to visualize the results")
                                    ], className="mb-3"),
                                    html.P([
                                        html.Strong("Tip: "),
                                        "Hover over the ",
                                        html.I(className="bi bi-question-circle"),
                                        " icons for detailed explanations of each parameter."
                                    ], className="mb-0")
                                ], color="info", className="text-start", id="welcome-message"),
                            ], id='graph-output-container', style={"minHeight": "400px"})
                        )
                    ])
                ])
            ], lg=8, md=7, className="mb-4"),
            
            # Right Column - Controls
            dbc.Col([
                basic_params_card,
                scheduling_params_card,
                timing_params_card,
                per_stage_timing_card,
                
                # Generate button with better styling
                dbc.Button([
                    html.I(className="bi bi-play-fill me-2"),
                    "Generate Schedules"
                ], 
                id='generate-button',
                color="primary",
                size="lg",
                className="w-100 mt-3",
                disabled=False
                )
            ], lg=4, md=5)
        ])
    ], fluid=True),
    
    # Toast container
    html.Div(id="toast-container", style={
        "position": "fixed", 
        "top": 20, 
        "right": 20, 
        "zIndex": 1050,
        "maxWidth": "400px"
    }),
    
    # Footer
    html.Footer([
        dbc.Container([
            html.Hr(className="mt-5"),
            dbc.Row([
                dbc.Col([
                    html.H6("About Pipeline Parallelism", className="text-muted mb-3"),
                    html.P([
                        "Pipeline parallelism is a distributed training technique that splits a model across multiple devices. ",
                        "This visualizer helps you understand different scheduling strategies and their performance characteristics."
                    ], className="small text-muted")
                ], md=4),
                dbc.Col([
                    html.H6("Scheduling Strategies", className="text-muted mb-3"),
                    html.Ul([
                        html.Li("1F1B: Standard pipeline with one forward, one backward", className="small text-muted"),
                        html.Li("ZB-1P: Zero bubble optimization", className="small text-muted"),
                        html.Li("Interleave: Virtual pipeline stages", className="small text-muted"),
                        html.Li("Overlap: Concurrent forward/backward", className="small text-muted")
                    ], className="list-unstyled")
                ], md=4),
                dbc.Col([
                    html.H6("Resources", className="text-muted mb-3"),
                    html.Div([
                        html.A([
                            html.I(className="bi bi-github me-2"),
                            "View on GitHub"
                        ], href="https://github.com/Victarry/PP-Schedule-Visualization", className="small text-muted d-block mb-2"),
                        html.A([
                            html.I(className="bi bi-question-circle me-2"),
                            "Report an Issue"
                        ], href="https://github.com/Victarry/PP-Schedule-Visualization/issues", className="small text-muted d-block")
                    ])
                ], md=4)
            ]),
            html.Hr(),
            html.P([
                "© 2024 Pipeline Parallelism Schedule Visualizer. ",
                "Built with ",
                html.I(className="bi bi-heart-fill text-danger"),
                " using Dash and Plotly."
            ], className="text-center text-muted small mb-3")
        ], fluid=True)
    ], className="mt-5 bg-light py-4")
])

# Keep the existing store for backward compatibility
app.layout.children.append(dcc.Store(id='advanced-timing-switch', data=False))

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
    Output('microbatch_group_size_per_vp_stage', 'invalid'),
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
    Input('microbatch_group_size_per_vp_stage', 'value'),
    Input('selected-strategies-store', 'data'), # Validate strategy selection
    prevent_initial_call=True # Prevent callback running on page load before user interaction
)
def validate_inputs(num_devices, num_stages, num_batches, p2p_latency,
                    op_time_forward, op_time_backward, op_time_backward_d, op_time_backward_w,
                    op_time_overlapped_fwd_bwd, microbatch_group_size_per_vp_stage, selected_strategies):
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
        "microbatch_group_size_per_vp_stage": microbatch_group_size_per_vp_stage is not None and (microbatch_group_size_per_vp_stage < 1 or microbatch_group_size_per_vp_stage % 1 != 0),
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
        is_invalid["microbatch_group_size_per_vp_stage"],
        strategy_feedback # Update strategy feedback based on validation
    )

# --- Callback to toggle Advanced Options Collapse ---
@app.callback(
    Output("advanced-timing-collapse", "is_open"),
    Input("advanced-timing-toggle", "n_clicks"),
    State("advanced-timing-collapse", "is_open"),
    prevent_initial_call=True,
)
def toggle_advanced_options(n_clicks, is_open):
    if n_clicks:
        return not is_open
    return is_open

# --- Callback to toggle Per-Stage Timing Collapse ---
@app.callback(
    Output("per-stage-timing-collapse", "is_open"),
    Input("per-stage-timing-toggle", "n_clicks"),
    State("per-stage-timing-collapse", "is_open"),
    prevent_initial_call=True,
)
def toggle_per_stage_timing(n_clicks, is_open):
    if n_clicks:
        return not is_open
    return is_open

# --- Callback to dynamically generate per-stage timing inputs ---
@app.callback(
    Output("per-stage-inputs-container", "children"),
    Input("num_stages", "value"),
)
def generate_per_stage_inputs(num_stages):
    if num_stages is None or num_stages < 1:
        return []
    
    # Limit to reasonable number of stages for UI
    num_stages = min(int(num_stages), 32)
    
    stage_inputs = []
    for stage_id in range(num_stages):
        stage_inputs.append(
            dbc.Row([
                dbc.Col([
                    html.Strong(f"Stage {stage_id}", className="text-muted")
                ], width=2, className="d-flex align-items-center"),
                dbc.Col([
                    dbc.InputGroup([
                        dbc.InputGroupText("F", style={"minWidth": "30px"}),
                        dbc.Input(
                            id={"type": "stage-forward", "index": stage_id},
                            type="number",
                            placeholder="1.0",
                            min=0.01,
                            step=0.01,
                            size="sm"
                        ),
                    ], size="sm")
                ], width=5),
                dbc.Col([
                    dbc.InputGroup([
                        dbc.InputGroupText("B", style={"minWidth": "30px"}),
                        dbc.Input(
                            id={"type": "stage-backward", "index": stage_id},
                            type="number",
                            placeholder="1.0",
                            min=0.01,
                            step=0.01,
                            size="sm"
                        ),
                    ], size="sm")
                ], width=5),
            ], className="mb-2 g-2")
        )
    
    # Add header row
    header = dbc.Row([
        dbc.Col([html.Small("Stage", className="text-muted fw-bold")], width=2),
        dbc.Col([html.Small("Forward Time", className="text-muted fw-bold")], width=5),
        dbc.Col([html.Small("Backward Time", className="text-muted fw-bold")], width=5),
    ], className="mb-2")
    
    return [header] + stage_inputs

# --- Client-side Callback for Strategy Card Selection ---
app.clientside_callback(
    """
    function(n_clicks_list, current_strategies) {
        const ctx = dash_clientside.callback_context;
        if (!ctx.triggered || ctx.triggered.length === 0) {
            return [dash_clientside.no_update, dash_clientside.no_update];
        }
        
        const triggered = ctx.triggered[0];
        const clickedIndex = JSON.parse(triggered.prop_id.split('.')[0]).index;
        
        let newStrategies = current_strategies ? [...current_strategies] : [];
        
        if (newStrategies.includes(clickedIndex)) {
            newStrategies = newStrategies.filter(s => s !== clickedIndex);
        } else {
            newStrategies.push(clickedIndex);
        }
        
        // Update card classes
        const allStrategies = ['1f1b', 'zb1p', '1f1b_overlap', '1f1b_interleave', '1f1b_interleave_overlap', 'dualpipe'];
        const cardClasses = allStrategies.map(strategy => 
            newStrategies.includes(strategy) 
                ? 'strategy-card p-3 text-center selected' 
                : 'strategy-card p-3 text-center'
        );
        
        return [newStrategies, cardClasses];
    }
    """,
    Output('selected-strategies-store', 'data'),
    Output({'type': 'strategy-card', 'index': ALL}, 'className'),
    Input({'type': 'strategy-card', 'index': ALL}, 'n_clicks'),
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
    State('microbatch_group_size_per_vp_stage', 'value'),
    State('selected-strategies-store', 'data'),
    State({'type': 'stage-forward', 'index': ALL}, 'value'),
    State({'type': 'stage-backward', 'index': ALL}, 'value'),
    prevent_initial_call=True
)
def update_graph(n_clicks, num_devices, num_stages, num_batches, p2p_latency,
                 op_time_forward, op_time_backward, op_time_backward_d, op_time_backward_w,
                 op_time_overlapped_fwd_bwd, microbatch_group_size_per_vp_stage,
                 selected_strategies, stage_forward_values, stage_backward_values):

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

                # Check if per-stage timing values are provided
                has_per_stage_forward = stage_forward_values and any(v is not None for v in stage_forward_values)
                has_per_stage_backward = stage_backward_values and any(v is not None for v in stage_backward_values)
                
                # Build forward timing - either per-stage dict or global value
                if has_per_stage_forward:
                    forward_times = {}
                    for stage_id in range(current_num_stages):
                        if stage_id < len(stage_forward_values) and stage_forward_values[stage_id] is not None:
                            forward_times[stage_id] = float(stage_forward_values[stage_id]) * time_scale_factor
                        else:
                            # Use global value as fallback (default 1.0 if not specified)
                            forward_times[stage_id] = float(op_time_forward if op_time_forward else 1.0) * time_scale_factor
                    op_times = {"forward": forward_times}
                else:
                    op_times = {"forward": float(op_time_forward) * time_scale_factor}

                # Build backward timing
                if split_backward:
                    op_times["backward_D"] = float(op_time_backward_d) * time_scale_factor
                    op_times["backward_W"] = float(op_time_backward_w) * time_scale_factor
                    op_times["backward"] = (float(op_time_backward_d) + float(op_time_backward_w)) * time_scale_factor
                else:
                    if has_per_stage_backward:
                        backward_times = {}
                        for stage_id in range(current_num_stages):
                            if stage_id < len(stage_backward_values) and stage_backward_values[stage_id] is not None:
                                backward_times[stage_id] = float(stage_backward_values[stage_id]) * time_scale_factor
                            else:
                                # Use global value as fallback (default 1.0 if not specified)
                                backward_times[stage_id] = float(op_time_backward if op_time_backward else 1.0) * time_scale_factor
                        op_times["backward"] = backward_times
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
                    microbatch_group_size_per_vp_stage=int(microbatch_group_size_per_vp_stage) if microbatch_group_size_per_vp_stage is not None else None,
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
    
    # --- Generate Graphs with improved layout ---
    if valid_results:
        max_execution_time = max(schedule.get_total_execution_time() for _, schedule, _ in valid_results)
        sorted_valid_results = sorted(valid_results, key=lambda x: strategy_display_order.index(x[0]) if x[0] in strategy_display_order else float('inf'))

        # Create tabs for multiple strategies
        tabs = []
        tab_panels = []
        
        for idx, (strategy, _, vis_data) in enumerate(sorted_valid_results):
            strategy_info = STRATEGY_INFO[strategy]
            
            # Create figure
            fig = create_pipeline_figure(vis_data, max_time=max_execution_time, show_progress=False)
            margin = max_execution_time * 0.05
            fig.update_layout(
                xaxis=dict(range=[0, max_execution_time + margin]),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(family="Inter, sans-serif"),
                height=400,  # Set explicit height
                autosize=True,  # Enable autosize
                margin=dict(l=60, r=20, t=40, b=60),  # Set proper margins
            )
            
            # Create tab
            tab_label = strategy_info['name']
            tab_value = f"tab-{strategy}"
            
            tabs.append(
                dbc.Tab(
                    label=tab_label,
                    tab_id=tab_value,
                    activeTabClassName="fw-bold"
                )
            )
            
            # Create tab panel content
            tab_panels.append(
                html.Div([
                    dbc.Alert([
                        html.I(className=f"{strategy_info['icon']} me-2"),
                        strategy_info['description']
                    ], color="light", className="mb-3"),
                    html.Div([
                        dcc.Graph(
                            figure=fig, 
                            config={
                                'displayModeBar': False,
                                'responsive': True  # Make graph responsive
                            },
                            className="dash-graph"
                        )
                    ], className="graph-container")
                ], 
                id=f"content-{strategy}",
                style={"display": "block" if idx == 0 else "none"}  # Show first tab by default
                )
            )
        
        # Create tabbed interface with callback to switch content
        graph_components = [
            dbc.Tabs(
                tabs,
                id="strategy-tabs",
                active_tab=f"tab-{sorted_valid_results[0][0]}",
                className="mb-3"
            ),
            html.Div(tab_panels, id="tab-content-container")
        ]

    # If there are graphs, use the component list, otherwise show a message
    output_content = []
    if graph_components:
        output_content = graph_components
    elif not toast_components:
        output_content = dbc.Alert([
            html.I(className="bi bi-info-circle-fill me-2"),
            "Click 'Generate Schedules' to visualize pipeline parallelism strategies."
        ], color="info", className="text-center")

    # Add execution time summary with improved design
    if execution_times:
        sorted_times = sorted(execution_times, key=lambda x: x[1])
        min_time = sorted_times[0][1] if sorted_times else None
        
        # Create metric cards for top strategies
        metric_cards = []
        for i, (strategy, time) in enumerate(sorted_times[:3]):  # Show top 3
            strategy_info = STRATEGY_INFO[strategy]
            badge_color = "success" if i == 0 else "primary" if i == 1 else "secondary"
            
            metric_cards.append(
                dbc.Col([
                    html.Div([
                        html.Div([
                            html.I(className=f"{strategy_info['icon']} mb-2", 
                                  style={"fontSize": "2rem", "color": "#667eea"}),
                            html.H3(f"{time:.2f} ms", className="metric-value"),
                            html.P(strategy_info['name'], className="metric-label mb-0"),
                            dbc.Badge(f"#{i+1}", color=badge_color, className="mt-2")
                        ], className="metric-card")
                    ])
                ], lg=4, md=6, className="mb-3")
            )
        
        # Create detailed comparison table
        table_rows = []
        for strategy, time in sorted_times:
            strategy_info = STRATEGY_INFO[strategy]
            efficiency = (min_time / time * 100) if min_time else 100
            
            table_rows.append(
                html.Tr([
                    html.Td([
                        html.I(className=f"{strategy_info['icon']} me-2"),
                        strategy_info['name']
                    ]),
                    html.Td(f"{time:.2f} ms"),
                    html.Td([
                        dbc.Progress(
                            value=efficiency,
                            color="success" if efficiency >= 95 else "warning" if efficiency >= 80 else "danger",
                            className="mb-0",
                            style={"height": "10px"}
                        ),
                        html.Small(f"{efficiency:.1f}%", className="ms-2 text-muted")
                    ])
                ], className="align-middle")
            )
        
        execution_summary = html.Div([
            html.H4([
                html.I(className="bi bi-speedometer2 section-icon"),
                "Performance Summary"
            ], className="section-title mt-5"),
            
            # Metric cards
            dbc.Row(metric_cards, className="mb-4"),
            
            # Detailed table
            dbc.Card([
                dbc.CardBody([
                    html.H5("Detailed Comparison", className="mb-3"),
                    dbc.Table([
                        html.Thead([
                            html.Tr([
                                html.Th("Strategy"),
                                html.Th("Execution Time"),
                                html.Th("Relative Efficiency", style={"width": "40%"})
                            ])
                        ]),
                        html.Tbody(table_rows)
                    ], hover=True, responsive=True, className="mb-0")
                ])
            ])
        ], className="execution-summary")
        
        # Append the execution summary
        if isinstance(output_content, list):
            output_content.append(execution_summary)
        else:
            output_content = [output_content, execution_summary]

    # Return graph components and toast components
    return output_content, toast_components

# --- Client-side Callback for Tab Switching ---
app.clientside_callback(
    """
    function(activeTab) {
        if (!activeTab) return window.dash_clientside.no_update;
        
        // Extract strategy from tab id (format: "tab-strategy")
        const activeStrategy = activeTab.replace("tab-", "");
        
        // Get all tab content divs
        const contentDivs = document.querySelectorAll('[id^="content-"]');
        
        contentDivs.forEach(div => {
            const strategy = div.id.replace("content-", "");
            if (strategy === activeStrategy) {
                div.style.display = "block";
            } else {
                div.style.display = "none";
            }
        });
        
        return window.dash_clientside.no_update;
    }
    """,
    Output("tab-content-container", "children"),  # Dummy output
    Input("strategy-tabs", "active_tab"),
    prevent_initial_call=True
)

# For Hugging Face Spaces deployment
server = app.server

if __name__ == '__main__':
    app.run_server(debug=False, host='0.0.0.0', port=7860) 