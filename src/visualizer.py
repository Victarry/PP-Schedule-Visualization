import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import argparse
from typing import List, Dict, Literal, Optional
from tqdm import tqdm
import base64

from src.execution_model import Schedule


def convert_schedule_to_visualization_format(schedule: Schedule):
    """
    Converts a Schedule object to the format needed for visualization.
    
    Returns:
        Dict[int, List[Dict]]: Dictionary mapping device_id to a list of operation dictionaries
    """
    # Make sure all operations have start and end times
    for op in schedule.ops.values():
        if op.start_time is None or op.end_time is None:
            raise ValueError("Operations must have start and end times. Run ScheduleExecutor.execute() first.")
    
    visualization_data = {}
    
    # Organize operations by device
    for device_id, device_queue in enumerate(schedule.dev_queues):
        visualization_data[device_id] = []
        
        for op in device_queue.ops:
            visualization_data[device_id].append({
                "type": op.op_type,
                "batch": op.batch_id + 1, # +1 because batch_id is 0-indexed
                "stage": op.stage_id,
                "start_time": op.start_time,
                "duration": op.end_time - op.start_time
            })
    
    return visualization_data


def create_pipeline_figure(schedule_data: Dict[int, List[Dict]], max_time=None, show_progress=True):
    """
    Create a Plotly figure for pipeline parallelism scheduling.

    Args:
        schedule_data: Dictionary mapping device IDs to lists of tasks (converted from Schedule)
        max_time: Optional maximum time to display
        show_progress: Whether to show a progress bar
    """
    # Find the number of devices
    num_devices = len(schedule_data)

    empty_color = "whitesmoke"
    # Colors for task types
    def get_color(op_type: str, stage_id: int):
        # Base colors
        forward_base_color = "royalblue"
        backward_base_color = "lightgreen"  # Changed from sandybrown to match your visualization
        
        virtual_stage = stage_id // num_devices

        if op_type == "forward":
            if virtual_stage == 0:
                return forward_base_color
            else:
                # Lighter shade for virtual_stage > 0
                return "lightskyblue"
        elif op_type == "backward":
            if virtual_stage == 0:
                return backward_base_color
            else:
                # Lighter shade for virtual_stage > 0
                return "lightseagreen"
        else:
            raise ValueError(f"Invalid operation type: {op_type}")

    # Find the maximum time in the schedule if not provided
    if max_time is None:
        max_time = 0
        for device in schedule_data:
            for task in schedule_data[device]:
                end_time = task["start_time"] + task["duration"]
                if end_time > max_time:
                    max_time = end_time

    # Create a figure
    fig = go.Figure()

    # Initialize progress tracking
    total_tasks = sum(len(tasks) for tasks in schedule_data.values())
    tasks_processed = 0

    if show_progress:
        progress_bar = tqdm(total=total_tasks + num_devices + 3, desc="Creating visualization")

    # Create a custom y-axis with no gaps between devices
    y_spacing = 1.0  # Use 1.0 for no gaps

    # Add rectangles for each task
    for device_idx, device in enumerate(schedule_data):
        device_idx_reversed = num_devices - device_idx - 1
        
        # Sort tasks by start time to ensure correct rendering
        sorted_tasks = sorted(schedule_data[device], key=lambda t: t["start_time"])
        
        for task in sorted_tasks:
            # Determine task color and text color
            if task["type"] == "forward":
                color = get_color(task["type"], task["stage"])
                text_color = "white"
                name = "Forward"
            elif task["type"] == "backward":
                color = get_color(task["type"], task["stage"])
                text_color = "black"
                name = "Backward"
            else:
                color = empty_color
                text_color = "black"
                name = "Unknown"

            # Add rectangle for the task
            start_time = task["start_time"]
            duration = task["duration"]
            
            # Calculate y positions with no gaps
            y_pos = device_idx_reversed * y_spacing
            
            # Create rectangle using shape
            fig.add_shape(
                type="rect",
                x0=start_time,
                y0=y_pos - 0.5,
                x1=start_time + duration,
                y1=y_pos + 0.5,
                line=dict(color="black", width=0.5),
                fillcolor=color,
                layer="above",
            )
            
            # Add batch number text
            fig.add_annotation(
                x=start_time + duration / 2,
                y=y_pos,
                text=f"{task['batch']}",  # Only show batch ID
                showarrow=False,
                font=dict(color=text_color, size=12, family="Arial, bold"),  # Increased font size
            )
            
            # Add hover data with additional details
            fig.add_trace(go.Scatter(
                x=[start_time + duration / 2],
                y=[y_pos],
                mode='markers',
                marker=dict(opacity=0),  # Invisible marker
                hoverinfo='text',
                text=f"Batch: {task['batch']}<br>Stage: {task['stage']}<br>Type: {name}<br>Start: {task['start_time']:.2f}<br>End: {task['start_time'] + task['duration']:.2f}<br>Duration: {task['duration']:.2f}",
                showlegend=False
            ))
            
            # Update progress
            if show_progress:
                tasks_processed += 1
                progress_bar.update(1)

    # Add custom legend
    legend_items = [
        dict(name="Forward", color=get_color("forward", 0)),
        dict(name="Backward", color=get_color("backward", 0)),
    ]
    
    for i, item in enumerate(legend_items):
        fig.add_trace(go.Scatter(
            x=[None],
            y=[None],
            mode='markers',
            marker=dict(size=10, color=item['color']),
            name=item['name'],
            showlegend=True
        ))
        if show_progress and i < len(legend_items) - 1:
            progress_bar.update(1)

    # Set axis properties
    device_labels = [f"Device {i}" for i in range(num_devices)]
    device_labels.reverse()  # Reverse to put Device 0 at the top
    
    # Calculate tick positions with no gaps
    tick_positions = [(num_devices - i - 1) * y_spacing for i in range(num_devices)]
    
    # Adjust the range to ensure there are no empty spaces at the end
    x_end = max_time * 1.05  # Add a small margin

    fig.update_layout(
        yaxis=dict(
            tickmode="array",
            tickvals=tick_positions,
            ticktext=device_labels,
            showgrid=False,
            zeroline=False,
        ),
        margin=dict(l=50, r=20, t=40, b=40),
        plot_bgcolor="white",
        title=dict(
            text="Pipeline Parallelism Schedule",
            x=0.5,
            y=0.98,  # Move title position closer to the top
            font=dict(size=20)
        ),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.1,  # Position below the plot
            xanchor="center",
            x=0.5
        ),
        width=1600, 
        height=400,  # Reduce height to make the visualization more compact
        bargap=0,
        bargroupgap=0,
    )

    if show_progress:
        progress_bar.update(1)
        progress_bar.close()

    return fig


def create_dash_app(schedule: Schedule, schedule_type="1f1b"):
    """
    Create a Dash app to visualize the pipeline schedule.
    
    Args:
        schedule: Schedule object to visualize
        schedule_type: Type of schedule ("1f1b" or other)
    """
    # Convert schedule to visualization format
    schedule_data = convert_schedule_to_visualization_format(schedule)
    
    # Create the app
    app = dash.Dash(__name__, title=f"Pipeline Parallelism Visualizer - {schedule_type}")
    
    app.layout = html.Div([
        html.H1(f"Pipeline Parallelism Visualizer - {schedule_type}", style={'textAlign': 'center'}),
        
        html.Div([
            html.Div([
                html.H3("Schedule Configuration:"),
                html.Ul([
                    html.Li(f"Number of devices: {schedule.config.num_devices}"),
                    html.Li(f"Number of stages: {schedule.config.num_stages}"),
                    html.Li(f"Number of batches: {schedule.config.num_batches}"),
                ]),
            ], className="config-section"),
            
            html.Button("Download Image", id="btn-download", 
                        style={
                            'marginTop': '20px',
                            'padding': '10px',
                            'backgroundColor': '#007BFF',
                            'color': 'white',
                            'border': 'none',
                            'borderRadius': '5px',
                            'cursor': 'pointer'
                        }),
                        
            dcc.Download(id="download-image"),
        ], style={'margin': '20px'}),
        
        html.Div(id="graph-container", children=[]),
        
        dcc.Graph(
            id="pipeline-graph",
            config={'displayModeBar': True, 'toImageButtonOptions': {'format': 'png', 'filename': 'pipeline_visualization'}}
        ),
    ])
    
    @app.callback(
        Output("pipeline-graph", "figure"),
        Input("graph-container", "children"),
        prevent_initial_call=False,
    )
    def load_graph(_):
        # Create the figure when the app loads
        return create_pipeline_figure(schedule_data, show_progress=True)
        
    @app.callback(
        Output("download-image", "data"),
        Input("btn-download", "n_clicks"),
        prevent_initial_call=True,
    )
    def download_image(n_clicks):
        # Generate the figure for download
        fig = create_pipeline_figure(schedule_data, show_progress=True)
        
        # Convert to base64 image
        img_bytes = fig.to_image(format="png", width=1600, height=1000, scale=2)
        img_base64 = base64.b64encode(img_bytes).decode('ascii')
        
        # Return the download data
        return dict(
            content=img_base64,
            filename=f"pipeline_visualization_{schedule_type}.png",
            type="image/png",
            base64=True
        )
    
    return app


def visualize_pipeline_parallelism_dash(
    schedule: Schedule,
    port: int = 8050,
    debug: bool = False
):
    """
    Launch a Dash app to visualize the pipeline schedule interactively.
    
    Args:
        schedule: Schedule object to visualize
        port: Port to run the Dash app on
        debug: Whether to run the Dash app in debug mode
    """
    app = create_dash_app(schedule)
    print(f"Starting Dash app on http://localhost:{port}/")
    app.run_server(debug=debug, port=port)


def save_pipeline_visualization_plotly(
    schedule: Schedule,
    output_file: str = "pipeline_visualization_plotly.png",
):
    """
    Save a static image of the pipeline schedule visualization.
    
    Args:
        schedule: Schedule object to visualize
        output_file: Path to save the image to
    """
    schedule_data = convert_schedule_to_visualization_format(schedule)
    fig = create_pipeline_figure(schedule_data, show_progress=True)
    
    print(f"Saving visualization to {output_file}...")
    fig.write_image(output_file, width=1600, height=400, scale=2)
    print(f"Visualization saved to {output_file}")

