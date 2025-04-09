import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
from typing import List, Dict
from tqdm import tqdm
from functools import lru_cache
import webbrowser
from threading import Timer

from src.execution_model import Schedule, OverlappedOperation


def convert_schedule_to_visualization_format(schedule: Schedule):
    """
    Converts a Schedule object to the format needed for visualization.

    Returns:
        Dict[int, List[Dict]]: Dictionary mapping device_id to a list of operation dictionaries
    """
    # Make sure all operations have start and end times
    for op in schedule.ops.values():
        if op.start_time is None or op.end_time is None:
            raise ValueError(
                "Operations must have start and end times. Run ScheduleExecutor.execute() first."
            )

    visualization_data = {}

    # Organize operations by device
    for device_id, device_queue in enumerate(schedule.device_queues):
        visualization_data[device_id] = []

        for op in device_queue.ops:
            # Handle both regular Operations and OverlappedOperations
            if isinstance(op, OverlappedOperation):
                visualization_data[device_id].append(
                    {
                        "type": op.op_type,
                        "batch": op.batch_id + 1,  # +1 because batch_id is 0-indexed
                        "stage": op.stage_id,
                        "start_time": op.start_time,
                        "duration": op.end_time - op.start_time,
                        "is_overlapped": True,
                        "operations": [
                            {
                                "type": nested_op.op_type,
                                "batch": nested_op.batch_id + 1,
                                "stage": nested_op.stage_id
                            }
                            for nested_op in op.operations
                        ]
                    }
                )
            else:
                visualization_data[device_id].append(
                    {
                        "type": op.op_type,
                        "batch": op.batch_id + 1,  # +1 because batch_id is 0-indexed
                        "stage": op.stage_id,
                        "start_time": op.start_time,
                        "duration": op.end_time - op.start_time,
                        "is_overlapped": False
                    }
                )

    return visualization_data


# Cache the color calculation as it's repeatedly called with the same parameters
@lru_cache(maxsize=128)
def get_color(op_type: str, stage_id: int, num_devices: int):
    # A more harmonious blue palette with low saturation and high brightness
    forward_colors = [
        "#0a5aff",  # Intense blue
        "#4c88ff",  # Blue (deeper)
        "#7aa7ff",  # Medium blue
        "#a8c5ff",  # Soft blue
        "#d6e4ff",  # Very light blue
    ]

    # Orange palette for backward operations with low saturation and high brightness
    backward_colors = [
        "#f47b00",  # Intense orange
        "#ffa952",  # Orange
        "#ffc78e",  # Light orange
        "#ffe6cc",  # Very light orange
    ]

    # Improved teal/turquoise palette with low saturation and high brightness
    backward_d_colors = [
        "#4dcccc",  # Light teal
        "#33b3b3",  # Teal
        "#009999",  # Medium teal
        "#008080",  # Dark teal
    ]

    # Improved green palette with low saturation and high brightness
    backward_w_colors = [
        "#33b373",  # Medium green
        "#009959",  # Forest green
        "#008040",  # Dark green
    ]
    
    virtual_stage = stage_id // num_devices

    # If virtual_stage is beyond our color list, cycle through the colors
    color_index = virtual_stage % len(forward_colors)

    if op_type == "forward":
        return forward_colors[color_index]
    elif op_type == "backward":
        return backward_colors[color_index % len(backward_colors)]
    elif op_type == "backward_D":
        return backward_d_colors[color_index % len(backward_d_colors)]
    elif op_type == "backward_W":
        return backward_w_colors[color_index % len(backward_w_colors)]
    else:
        raise ValueError(f"Invalid operation type: {op_type}")


def create_pipeline_figure(
    schedule_data: Dict[int, List[Dict]], max_time=None, show_progress=True
):
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
    
    # Determine maximum batch number to decide whether to show text labels
    max_batch = 0
    for device in schedule_data:
        for task in schedule_data[device]:
            max_batch = max(max_batch, task["batch"])
    
    # Flag to determine whether to show text labels
    num_operations_per_device = len(schedule_data[0])
    show_text_labels = num_operations_per_device <= 64

    # Create a figure
    fig = go.Figure()

    # Initialize progress tracking
    total_tasks = sum(len(tasks) for tasks in schedule_data.values())
    tasks_processed = 0

    if show_progress:
        progress_bar = tqdm(
            total=total_tasks + num_devices + 3, desc="Creating visualization"
        )

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
            # Calculate y positions with no gaps
            y_pos = device_idx_reversed * y_spacing
            start_time = task["start_time"]
            duration = task["duration"]
            
            # Special handling for overlapped operations
            if task.get("is_overlapped", False) and "operations" in task:
                # Prepare hover text for the entire overlapped operation
                op_details = "<br>".join([
                    f"- {op['type']} (Batch {op['batch']}, Stage {op['stage']})"
                    for op in task["operations"]
                ])
                hover_text = (
                    f"Overlapped Operations:<br>{op_details}<br>"
                    f"Start: {task['start_time']:.2f}<br>"
                    f"End: {task['start_time'] + task['duration']:.2f}<br>"
                    f"Duration: {task['duration']:.2f}"
                )
                
                # Add invisible marker for hover info
                hover_traces.append(
                    dict(
                        x=[start_time + duration / 2],
                        y=[y_pos],
                        mode="markers",
                        marker=dict(opacity=0),  # Invisible marker
                        hoverinfo="text",
                        text=hover_text,
                        showlegend=False,
                    )
                )
                
                # Calculate height of each sub-operation
                sub_height = 1.0 / len(task["operations"])
                
                # Add rectangles and annotations for each sub-operation
                for i, sub_op in enumerate(task["operations"]):
                    # Determine color for this sub-operation
                    color = get_color(sub_op["type"], sub_op["stage"], num_devices)
                    
                    # Calculate y position for this sub-operation
                    sub_y_pos_bottom = y_pos - 0.5 + (i * sub_height)
                    sub_y_pos_top = sub_y_pos_bottom + sub_height
                    sub_y_center = (sub_y_pos_bottom + sub_y_pos_top) / 2
                    
                    # Add rectangle for this sub-operation
                    shapes.append(
                        dict(
                            type="rect",
                            x0=start_time,
                            y0=sub_y_pos_bottom,
                            x1=start_time + duration,
                            y1=sub_y_pos_top,
                            line=dict(color="black", width=0.5),
                            fillcolor=color,
                            layer="above",
                        )
                    )
                    
                    # Add batch number text for this sub-operation only if show_text_labels is True
                    if show_text_labels:
                        # Determine text color based on background color
                        if sub_op["type"] in ["backward", "backward_D", "backward_W"]:
                            text_color = "black"
                        else:
                            text_color = "white"
                            
                        annotations.append(
                            dict(
                                x=start_time + duration / 2,
                                y=sub_y_center,
                                text=f"{sub_op['batch']}",
                                showarrow=False,
                                font=dict(color=text_color, size=12, family="Arial, bold"),
                            )
                        )
            else:
                # Regular (non-overlapped) operation
                # Determine task color and text color
                if task["type"] == "forward":
                    color = get_color(task["type"], task["stage"], num_devices)
                    text_color = "white"
                    name = "Forward"
                elif task["type"] == "backward":
                    color = get_color(task["type"], task["stage"], num_devices)
                    text_color = "black"
                    name = "Backward"
                elif task["type"] == "backward_D":
                    color = get_color(task["type"], task["stage"], num_devices)
                    text_color = "black"
                    name = "Backward (Grad)"
                elif task["type"] == "backward_W":
                    color = get_color(task["type"], task["stage"], num_devices)
                    text_color = "black"
                    name = "Backward (Weight)"
                else:
                    color = empty_color
                    text_color = "black"
                    name = "Unknown"

                # Add rectangle for the task
                shapes.append(
                    dict(
                        type="rect",
                        x0=start_time,
                        y0=y_pos - 0.5,
                        x1=start_time + duration,
                        y1=y_pos + 0.5,
                        line=dict(color="black", width=0.5),
                        fillcolor=color,
                        layer="above",
                    )
                )

                # Add batch number text only if show_text_labels is True
                if show_text_labels:
                    annotations.append(
                        dict(
                            x=start_time + duration / 2,
                            y=y_pos,
                            text=f"{task['batch']}",
                            showarrow=False,
                            font=dict(color=text_color, size=12, family="Arial, bold"),
                        )
                    )

                # Prepare hover data
                hover_text = (
                    f"Batch: {task['batch']}<br>"
                    f"Stage: {task['stage']}<br>"
                    f"Type: {name}<br>"
                    f"Start: {task['start_time']:.2f}<br>"
                    f"End: {task['start_time'] + task['duration']:.2f}<br>"
                    f"Duration: {task['duration']:.2f}"
                )

                hover_traces.append(
                    dict(
                        x=[start_time + duration / 2],
                        y=[y_pos],
                        mode="markers",
                        marker=dict(opacity=0),  # Invisible marker
                        hoverinfo="text",
                        text=hover_text,
                        showlegend=False,
                    )
                )

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

    # Check if overlapped operations exist
    has_overlapped = any(
        task.get("is_overlapped", False)
        for device in schedule_data
        for task in schedule_data[device]
    )

    # Add forward and backward items for each virtual stage
    for vs in range(max_virtual_stage + 1):
        legend_items.append(
            dict(
                name=f"Forward (VS {vs})",
                color=get_color("forward", vs * num_devices, num_devices),
            )
        )
        legend_items.append(
            dict(
                name=f"Backward (VS {vs})",
                color=get_color("backward", vs * num_devices, num_devices),
            )
        )
        # Add entries for split backward operations if this is a zb1p schedule
        if any(
            task["type"] in ["backward_D", "backward_W"]
            for device in schedule_data
            for task in schedule_data[device]
        ):
            legend_items.append(
                dict(
                    name=f"Backward Grad (VS {vs})",
                    color=get_color("backward_D", vs * num_devices, num_devices),
                )
            )
            legend_items.append(
                dict(
                    name=f"Backward Weight (VS {vs})",
                    color=get_color("backward_W", vs * num_devices, num_devices),
                )
            )

    # If no tasks found, add default legend items
    if not legend_items:
        legend_items = [
            dict(name="Forward (VS 0)", color=get_color("forward", 0, num_devices)),
            dict(name="Backward (VS 0)", color=get_color("backward", 0, num_devices)),
            dict(
                name="Backward Grad (VS 0)",
                color=get_color("backward_D", 0, num_devices),
            ),
            dict(
                name="Backward Weight (VS 0)",
                color=get_color("backward_W", 0, num_devices),
            ),
        ]

    for i, item in enumerate(legend_items):
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                marker=dict(size=10, color=item["color"]),
                name=item["name"],
                showlegend=True,
            )
        )
        if show_progress and i < len(legend_items) - 1:
            progress_bar.update(1)

    # Set axis properties
    device_labels = [f"Device {i+1}" for i in range(num_devices)]

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
            font=dict(size=20),
        ),
        legend=dict(
            orientation="v",  # Changed from horizontal to vertical
            yanchor="top",
            y=1.02,  # Position at the top
            xanchor="right",
            x=1.20,  # Position further to the right to accommodate more items
            title=dict(text="<b>Operation Types:</b>"),
            itemsizing="constant",
            tracegroupgap=0,
        ),
        width=2000,  # Increase width to accommodate the expanded legend
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


def create_dash_app(
    schedule: Schedule, schedule_type="1f1b", enable_caching: bool = True
):
    """
    Create a Dash app to visualize the pipeline schedule.

    Args:
        schedule: Schedule object to visualize
        schedule_type: Type of schedule ("1f1b", "zb1p", or custom description)
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
    app.layout = html.Div(
        [
            html.H1(
                f"Pipeline Parallelism Visualization - {schedule_type}",
                style={"textAlign": "center"},
            ),
            html.Div(
                [
                    html.P(
                        f"Number of devices: {len(schedule_data)}",
                        style={"display": "inline-block", "marginRight": "20px"},
                    ),
                    html.P(
                        f"Total tasks: {total_tasks}",
                        style={"display": "inline-block", "marginRight": "20px"},
                    ),
                ],
                style={"marginBottom": "20px"},
            ),
            html.Div(id="graph-container", children=[]),
            dcc.Loading(
                id="loading-graph",
                type="circle",
                children=[
                    dcc.Graph(
                        id="pipeline-graph",
                        config={
                            "displayModeBar": True,
                            "toImageButtonOptions": {
                                "format": "png",
                                "filename": "pipeline_visualization",
                            },
                        },
                    ),
                ],
            ),
        ]
    )

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
    enable_caching: bool = True,
    schedule_type="1f1b",
    open_browser: bool = True,
):
    """
    Launch a Dash app to visualize the pipeline schedule interactively.

    Args:
        schedule: Schedule object to visualize
        port: Port to run the Dash app on
        debug: Whether to run the Dash app in debug mode
        enable_caching: Whether to cache schedule data and figures
        schedule_type: Type of schedule ("1f1b", "zb1p", or custom description)
        open_browser: Whether to automatically open a browser window
    """
    app = create_dash_app(
        schedule, schedule_type=schedule_type, enable_caching=enable_caching
    )
    
    # Define function to open browser after a short delay
    def open_browser_tab():
        webbrowser.open_new_tab(f"http://localhost:{port}/")
    
    # Open browser automatically if requested
    if open_browser:
        # Use a timer to open the browser after the server has started
        Timer(1.0, open_browser_tab).start()
    
    print(f"Starting Dash app on http://localhost:{port}/")
    app.run_server(debug=debug, port=port)
