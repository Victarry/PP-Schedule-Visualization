# Pipeline Parallelism Dash Visualizer

This is an interactive Dash-based visualizer for pipeline parallelism scheduling, complementing the existing Matplotlib-based visualization.

## Features

- **Static image generation** similar to the Matplotlib version
- **Interactive web-based visualization** with Dash
- **Download functionality** to save the visualization as PNG
- **Progress indication** during figure creation and image generation
- **Compatible API** with the existing visualizer

## Installation

Install the required dependencies:

```bash
pip install -r requirements-dash.txt
```

## Usage

### From Python

```python
from pipeline import create_1f1b_schedule
from dash_visualizer import visualize_pipeline_parallelism_dash, save_pipeline_visualization_plotly

# Create a schedule
schedule = create_1f1b_schedule(
    num_stages=4,
    num_batches=8,
    forward_times=[1.0, 1.0, 1.0, 1.0],
    backward_times=[2.0, 2.0, 2.0, 2.0],
)

# Generate a static image
save_pipeline_visualization_plotly(
    schedule=schedule,
    schedule_type="1f1b",
    output_file="pipeline_plotly.png"
)

# OR launch an interactive Dash app
visualize_pipeline_parallelism_dash(
    schedule=schedule,
    schedule_type="1f1b",
    port=8050,
    debug=False
)
```

### Using the Command Line

You can use the updated command line interface:

```bash
# Generate a static image with Dash/Plotly
python pipeline.py --visualizer dash --output-file pipeline_viz.png

# Launch an interactive Dash app
python pipeline.py --visualizer dash-interactive

# Use the original Matplotlib visualizer
python pipeline.py --visualizer matplotlib
```

You can also use the dash_visualizer.py script directly for testing:

```bash
# Generate a static image
python dash_visualizer.py --output test_viz.png

# Launch an interactive app
python dash_visualizer.py --interactive
```

## Differences from Matplotlib Visualizer

The Dash-based visualizer provides all the same visual elements as the Matplotlib version:
- Color-coded rectangles for forward, backward, and optimizer operations
- Batch numbers displayed inside each rectangle
- Device labels on the y-axis
- Clear legend

Additional features:
- Interactive web interface
- Hovering over elements to see details
- Download button to save the visualization
- Progress bars for tracking visualization creation
- Responsive layout that works well on different screen sizes 