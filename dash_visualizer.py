import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
import numpy as np
from typing import List, Dict, Literal
from tqdm import tqdm
import time


def create_pipeline_figure(schedule: Dict[int, List[Dict]], max_time=None, show_progress=True):
    """
    Create a Plotly figure for pipeline parallelism scheduling.

    Args:
        schedule: Dictionary mapping device IDs to lists of tasks.
                 Each task is a dictionary with keys:
                 - 'type': 'forward', 'backward', or 'optimizer'
                 - 'batch': batch number
                 - 'start_time': start time of the task
                 - 'duration': duration of the task
        max_time: Optional maximum time to display
        show_progress: Whether to show a progress bar
    """
    # Colors for task types
    forward_color = "royalblue"
    backward_color = "sandybrown"  
    optimizer_color = "#FFEFCF"
    empty_color = "whitesmoke"

    # Find the number of stages (devices)
    num_stages = len(schedule)

    # Find the maximum time in the schedule if not provided
    if max_time is None:
        max_time = 0
        for device in schedule:
            for task in schedule[device]:
                end_time = task["start_time"] + task["duration"]
                if end_time > max_time:
                    max_time = end_time

    # Create a figure
    fig = go.Figure()

    # Initialize progress tracking
    total_tasks = sum(len(tasks) for tasks in schedule.values())
    tasks_processed = 0

    if show_progress:
        progress_bar = tqdm(total=total_tasks + num_stages + 3, desc="Creating visualization")

    # Add background for empty cells
    for device_idx in range(num_stages):
        device_idx_reversed = num_stages - device_idx - 1  # Reverse for plotting
        fig.add_trace(go.Scatter(
            x=[0, max_time],
            y=[device_idx_reversed, device_idx_reversed],
            mode='lines',
            line=dict(color='lightgray', width=0.5),
            showlegend=False,
            hoverinfo='none'
        ))
        if show_progress:
            progress_bar.update(1)

    # Add rectangles for each task
    for device_idx, device in enumerate(schedule):
        device_idx_reversed = num_stages - device_idx - 1
        
        for task in schedule[device]:
            # Determine task color and text color
            if task["type"] == "forward":
                color = forward_color
                text_color = "white"
                name = "Forward"
            elif task["type"] == "backward":
                color = backward_color
                text_color = "black"
                name = "Backward"
            else:  # optimizer or any other type
                color = optimizer_color
                text_color = "black"
                name = "Optimizer step"
            
            # Add rectangle for the task
            start_time = task["start_time"]
            duration = task["duration"]
            
            # Create rectangle using shape
            fig.add_shape(
                type="rect",
                x0=start_time,
                y0=device_idx_reversed - 0.4,
                x1=start_time + duration,
                y1=device_idx_reversed + 0.4,
                line=dict(color="black", width=0.5),
                fillcolor=color,
                layer="above",
            )
            
            # Add batch number text
            fig.add_annotation(
                x=start_time + duration / 2,
                y=device_idx_reversed,
                text=str(task["batch"]),
                showarrow=False,
                font=dict(color=text_color, size=10, family="Arial, bold"),
            )
            
            # Update progress
            if show_progress:
                tasks_processed += 1
                progress_bar.update(1)

    # Add custom legend
    legend_items = [
        dict(name="Forward", color=forward_color),
        dict(name="Backward", color=backward_color),
        dict(name="Optimizer step", color=optimizer_color)
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
    device_labels = [f"Device {i+1}" for i in range(num_stages)]
    device_labels.reverse()  # Reverse to put Device 1 at the top
    
    fig.update_layout(
        xaxis=dict(
            showticklabels=False,
            showgrid=False,
            zeroline=False,
            title="Time →",
            range=[0, max_time + 0.5]
        ),
        yaxis=dict(
            tickmode="array",
            tickvals=list(range(num_stages)),
            ticktext=device_labels,
            showgrid=False,
            zeroline=False,
            range=[-0.5, num_stages - 0.5]
        ),
        margin=dict(l=50, r=50, t=50, b=50),
        plot_bgcolor="white",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.2,
            xanchor="center",
            x=0.5
        )
    )

    if show_progress:
        progress_bar.update(1)  # Final update for layout
        progress_bar.close()

    return fig


def create_dash_app(schedule: Dict[int, List[Dict]], schedule_type="1f1b"):
    """
    Create a Dash app for interactive visualization of pipeline scheduling.

    Args:
        schedule: Dictionary mapping device IDs to lists of tasks
        schedule_type: Type of scheduling algorithm used
    """
    app = dash.Dash(__name__, title="Pipeline Parallelism Visualization")
    
    app.layout = html.Div([
        html.H1(f"Pipeline Parallelism Visualization ({schedule_type.upper()})", 
                style={'textAlign': 'center'}),
        
        html.Div(id="loading-container", children=[
            dcc.Loading(
                id="loading-graph",
                type="circle",
                children=[
                    html.Div(id="graph-container", children=[
                        dcc.Graph(
                            id='pipeline-graph',
                            style={'height': '600px'}
                        )
                    ])
                ]
            )
        ]),
        
        html.Div([
            html.Button("Download PNG", id="btn-download", 
                      style={'margin': '10px'}),
            dcc.Download(id="download-image")
        ], style={'textAlign': 'center', 'marginTop': '20px'})
    ])
    
    @app.callback(
        Output("pipeline-graph", "figure"),
        Input("graph-container", "children"),
        prevent_initial_call=False,
    )
    def load_graph(_):
        # Create the figure when the app loads
        return create_pipeline_figure(schedule, show_progress=True)
    
    @app.callback(
        Output("download-image", "data"),
        Input("btn-download", "n_clicks"),
        prevent_initial_call=True,
    )
    def download_image(n_clicks):
        # Show progress in terminal for downloads
        fig = create_pipeline_figure(schedule, show_progress=True)
        img_bytes = fig.to_image(format="png", scale=3)
        return dict(
            content=img_bytes,
            filename="pipeline_visualization.png"
        )
    
    return app


def visualize_pipeline_parallelism_dash(
    schedule: Dict[int, List[Dict]],
    schedule_type: Literal["simple", "1f1b"] = "1f1b",
    port: int = 8050,
    debug: bool = False
):
    """
    Create an interactive Dash visualization for pipeline parallelism scheduling.

    Args:
        schedule: Dictionary mapping device IDs to lists of tasks
        schedule_type: Type of scheduling algorithm used ("simple" or "1f1b")
        port: Port number to run the Dash app
        debug: Whether to run the app in debug mode
    """
    app = create_dash_app(schedule, schedule_type)
    print(f"Starting Dash app on http://localhost:{port}/")
    app.run_server(debug=debug, port=port)


def save_pipeline_visualization_plotly(
    schedule: Dict[int, List[Dict]],
    schedule_type: Literal["simple", "1f1b"] = "1f1b",
    output_file: str = "pipeline_visualization_plotly.png",
):
    """
    Save a static Plotly visualization of pipeline parallelism scheduling.

    Args:
        schedule: Dictionary mapping device IDs to lists of tasks
        schedule_type: Type of scheduling algorithm used
        output_file: Path to save the visualization
    """
    print(f"Creating visualization for {len(schedule)} devices...")
    fig = create_pipeline_figure(schedule, show_progress=True)
    
    # Update layout for static image
    fig.update_layout(
        title=f"Pipeline Parallelism Visualization ({schedule_type.upper()})",
        title_x=0.5
    )
    
    print(f"Saving image to {output_file}...")
    # Save as image
    fig.write_image(output_file, scale=3)
    print(f"Visualization saved to {output_file}")


if __name__ == "__main__":
    # Example usage
    import argparse
    from pipeline import create_1f1b_schedule
    
    parser = argparse.ArgumentParser(description="Pipeline Parallelism Visualizer")
    parser.add_argument("--num-stages", type=int, default=4, help="Number of pipeline stages")
    parser.add_argument("--num-batches", type=int, default=8, help="Number of microbatches")
    parser.add_argument("--interactive", action="store_true", help="Run interactive Dash app")
    parser.add_argument("--port", type=int, default=8050, help="Port for Dash app")
    parser.add_argument("--output", type=str, default="pipeline_visualization_plotly.png", help="Output file for static image")
    args = parser.parse_args()
    
    # Create an example schedule
    forward_times = [1.0] * args.num_stages
    backward_times = [2.0] * args.num_stages
    
    schedule = create_1f1b_schedule(
        num_stages=args.num_stages,
        num_batches=args.num_batches,
        forward_times=forward_times,
        backward_times=backward_times,
    )
    
    if args.interactive:
        visualize_pipeline_parallelism_dash(schedule, port=args.port)
    else:
        save_pipeline_visualization_plotly(schedule, output_file=args.output) 