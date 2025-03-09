import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
from typing import List, Dict
from tqdm import tqdm
from functools import lru_cache

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


# Cache the color calculation as it's repeatedly called with the same parameters
@lru_cache(maxsize=128)
def get_color(op_type: str, stage_id: int, num_devices: int):
    # Color palettes for different virtual stages
    forward_colors = [
        "royalblue",      # Stage 0
        "lightskyblue",   # Stage 1
        "cornflowerblue", # Stage 2
        "steelblue",      # Stage 3
        "dodgerblue",     # Stage 4
        "deepskyblue",    # Stage 5
        "mediumblue",     # Stage 6
        "mediumslateblue",# Stage 7
        "slateblue",      # Stage 8
        "darkslateblue"   # Stage 9
    ]
    
    backward_colors = [
        "lightgreen",     # Stage 0
        "mediumseagreen", # Stage 1
        "seagreen",       # Stage 2
        "lightseagreen",  # Stage 3
        "mediumaquamarine", # Stage 4
        "mediumspringgreen", # Stage 5
        "springgreen",    # Stage 6
        "palegreen",      # Stage 7
        "limegreen",      # Stage 8
        "forestgreen"     # Stage 9
    ]

    virtual_stage = stage_id // num_devices

    # If virtual_stage is beyond our color list, cycle through the colors
    color_index = virtual_stage % len(forward_colors)

    if op_type == "forward":
        return forward_colors[color_index]
    elif op_type == "backward":
        return backward_colors[color_index]
    else:
        raise ValueError(f"Invalid operation type: {op_type}")


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

    # Batch processing for increased performance
    shapes = []
    annotations = []
    hover_traces = []

    # Add rectangles for each task
    for device_idx, device in enumerate(schedule_data):
        device_idx_reversed = num_devices - device_idx - 1
        
        # Sort tasks by start time to ensure correct rendering
        sorted_tasks = sorted(schedule_data[device], key=lambda t: t["start_time"])
        
        for task in sorted_tasks:
            # Determine task color and text color
            if task["type"] == "forward":
                color = get_color(task["type"], task["stage"], num_devices)
                text_color = "white"
                name = "Forward"
            elif task["type"] == "backward":
                color = get_color(task["type"], task["stage"], num_devices)
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
            
            # Create rectangle using shape (batch-add later)
            shapes.append(dict(
                type="rect",
                x0=start_time,
                y0=y_pos - 0.5,
                x1=start_time + duration,
                y1=y_pos + 0.5,
                line=dict(color="black", width=0.5),
                fillcolor=color,
                layer="above",
            ))
            
            # Add batch number text (batch-add later)
            annotations.append(dict(
                x=start_time + duration / 2,
                y=y_pos,
                text=f"{task['batch']}",  
                showarrow=False,
                font=dict(color=text_color, size=12, family="Arial, bold"),
            ))
            
            # Prepare hover data (add traces in batches later)
            hover_text = f"Batch: {task['batch']}<br>Stage: {task['stage']}<br>Type: {name}<br>Start: {task['start_time']:.2f}<br>End: {task['start_time'] + task['duration']:.2f}<br>Duration: {task['duration']:.2f}"
            
            hover_traces.append(dict(
                x=[start_time + duration / 2],
                y=[y_pos],
                mode='markers',
                marker=dict(opacity=0),  # Invisible marker
                hoverinfo='text',
                text=hover_text,
                showlegend=False
            ))
            
            # Update progress
            if show_progress:
                tasks_processed += 1
                progress_bar.update(1)

    # Add all shapes at once for better performance
    fig.update_layout(shapes=shapes)
    
    # Add all annotations at once
    fig.update_layout(annotations=annotations)
    
    # Add all hover traces at once
    for trace in hover_traces:
        fig.add_trace(go.Scatter(**trace))

    # Add custom legend
    legend_items = []
    
    # Find the maximum virtual stage in the data
    max_virtual_stage = 0
    for device in schedule_data:
        for task in schedule_data[device]:
            virtual_stage = task["stage"] // num_devices
            max_virtual_stage = max(max_virtual_stage, virtual_stage)
    
    # Add forward and backward items for each virtual stage
    for vs in range(max_virtual_stage + 1):
        legend_items.append(dict(
            name=f"Forward (VS {vs})", 
            color=get_color("forward", vs * num_devices, num_devices)
        ))
        legend_items.append(dict(
            name=f"Backward (VS {vs})", 
            color=get_color("backward", vs * num_devices, num_devices)
        ))
    
    # If no tasks found, add default legend items
    if not legend_items:
        legend_items = [
            dict(name="Forward (VS 0)", color=get_color("forward", 0, num_devices)),
            dict(name="Backward (VS 0)", color=get_color("backward", 0, num_devices)),
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

    title_text = "Pipeline Parallelism Schedule"

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
            text=title_text,
            x=0.5,
            y=0.98,  # Move title position closer to the top
            font=dict(size=20)
        ),
        legend=dict(
            orientation="v",  # Changed from horizontal to vertical
            yanchor="top",
            y=1.02,  # Position at the top
            xanchor="right",
            x=1.15,   # Position to the right of the plot
            title=dict(text="<b>Operation Types:</b>"),
            itemsizing="constant",
            tracegroupgap=0
        ),
        width=1800,  # Increase width to accommodate the legend
        height=400,  # Maintain current height
        bargap=0,
        bargroupgap=0,
    )

    if show_progress:
        progress_bar.update(1)
        progress_bar.close()

    return fig


# Cache for storing processed schedule data
_schedule_data_cache = {}

def create_dash_app(schedule: Schedule, schedule_type="1f1b", enable_caching: bool = True):
    """
    Create a Dash app to visualize the pipeline schedule.
    
    Args:
        schedule: Schedule object to visualize
        schedule_type: Type of schedule ("1f1b" or custom description)
        enable_caching: Whether to cache the schedule data and figure
    """
    # Process schedule data only once and cache it
    global _schedule_data_cache
    cache_key = id(schedule)
    
    if enable_caching and cache_key in _schedule_data_cache:
        schedule_data = _schedule_data_cache[cache_key]
        print("Using cached schedule data")
    else:
        schedule_data = convert_schedule_to_visualization_format(schedule)
        if enable_caching:
            _schedule_data_cache[cache_key] = schedule_data
            print("Cached schedule data")
    
    total_tasks = sum(len(tasks) for tasks in schedule_data.values())
    print(f"Total tasks in schedule: {total_tasks}")

    app = dash.Dash(__name__)
    app.title = f"Pipeline Parallelism Visualization - {schedule_type}"

    # Create a more informative layout with data size information
    app.layout = html.Div([
        html.H1(f"Pipeline Parallelism Visualization - {schedule_type}", style={"textAlign": "center"}),
        
        html.Div([
            html.P(f"Number of devices: {len(schedule_data)}", style={"display": "inline-block", "marginRight": "20px"}),
            html.P(f"Total tasks: {total_tasks}", style={"display": "inline-block", "marginRight": "20px"}),
        ], style={"marginBottom": "20px"}),
        
        html.Div(id="graph-container", children=[]),
        
        dcc.Loading(
            id="loading-graph",
            type="circle",
            children=[
                dcc.Graph(
                    id="pipeline-graph",
                    config={'displayModeBar': True, 'toImageButtonOptions': {'format': 'png', 'filename': 'pipeline_visualization'}}
                ),
            ]
        ),
    ])
    
    # Cache for storing figure to avoid regenerating it
    figure_cache = {}
    
    @app.callback(
        Output("pipeline-graph", "figure"),
        Input("graph-container", "children"),
        prevent_initial_call=False,
    )
    def load_graph(_):
        # Use cached figure if available
        cache_key = f"{id(schedule)}"
        if enable_caching and cache_key in figure_cache:
            print("Using cached figure")
            return figure_cache[cache_key]
        
        # Create the figure
        figure = create_pipeline_figure(schedule_data, show_progress=True)
        
        # Cache the figure
        if enable_caching:
            figure_cache[cache_key] = figure
            print("Cached figure")
            
        return figure

    return app


def visualize_pipeline_parallelism_dash(
    schedule: Schedule,
    port: int = 8050,
    debug: bool = False,
    enable_caching: bool = True
):
    """
    Launch a Dash app to visualize the pipeline schedule interactively.
    
    Args:
        schedule: Schedule object to visualize
        port: Port to run the Dash app on
        debug: Whether to run the Dash app in debug mode
        enable_caching: Whether to cache schedule data and figures
    """
    app = create_dash_app(schedule, enable_caching=enable_caching)
    print(f"Starting Dash app on http://localhost:{port}/")
    app.run_server(debug=debug, port=port)
