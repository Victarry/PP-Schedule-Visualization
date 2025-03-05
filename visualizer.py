import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from typing import List, Dict, Literal


def visualize_pipeline_parallelism(
    schedule: Dict[int, List[Dict]],
    schedule_type: Literal["simple", "1f1b"] = "1f1b",
    output_file: str = "pipeline_visualization.png",
):
    """
    Visualize pipeline parallelism scheduling.

    Args:
        schedule: Dictionary mapping device IDs to lists of tasks.
                 Each task is a dictionary with keys:
                 - 'type': 'forward', 'backward', or 'optimizer'
                 - 'batch': batch number
                 - 'start_time': start time of the task
                 - 'duration': duration of the task
        schedule_type: Type of scheduling algorithm used ("simple" or "1f1b")
        output_file: Path to save the visualization
    """
    # Colors for task types
    forward_color = "royalblue"
    backward_color = "sandybrown"  # Changed to match the reference image
    optimizer_color = "#FFEFCF"    # Light beige for optimizer steps
    empty_color = "whitesmoke"     # Very light gray for empty cells

    # Find the number of stages (devices)
    num_stages = len(schedule)

    # Find the maximum time in the schedule
    max_time = 0
    for device in schedule:
        for task in schedule[device]:
            end_time = task["start_time"] + task["duration"]
            if end_time > max_time:
                max_time = end_time

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(15, 4))
    
    # Create an empty grid with light gray color
    for device_idx in range(num_stages):
        device_idx_reversed = num_stages - device_idx - 1  # Reverse the device index for plotting
        for t in range(int(max_time) + 1):
            rect = Rectangle(
                (t, device_idx_reversed),
                1.0,
                1.0,
                edgecolor="lightgray",
                facecolor=empty_color,
                linewidth=0.5,
            )
            ax.add_patch(rect)

    # Plot the schedule
    for device_idx, device in enumerate(schedule):
        device_idx_reversed = num_stages - device_idx - 1  # Reverse the device index for plotting
        for task in schedule[device]:
            # Determine task color
            if task["type"] == "forward":
                color = forward_color
                text_color = "white"
            elif task["type"] == "backward":
                color = backward_color
                text_color = "black"
            else:  # optimizer or any other type
                color = optimizer_color
                text_color = "black"
                
            rect = Rectangle(
                (task["start_time"], device_idx_reversed),
                task["duration"],
                1.0,
                edgecolor="black",
                facecolor=color,
                linewidth=0.5,
            )
            ax.add_patch(rect)

            # Add text (batch number)
            ax.text(
                task["start_time"] + task["duration"] / 2,
                device_idx_reversed + 0.5,
                str(task["batch"]),
                ha="center",
                va="center",
                fontsize=10,
                fontweight="bold",
                color=text_color,
            )

    # Set axis limits and labels
    ax.set_xlim(0, max_time + 0.5)
    ax.set_ylim(-0.5, num_stages + 0.5)
    ax.set_yticks(np.arange(num_stages) + 0.5)
    
    # Reverse the order: Device 1 at the top, highest number at the bottom
    device_labels = [f"Device {i+1}" for i in range(num_stages)]
    device_labels.reverse()  # Reverse to put Device 1 at the top
    ax.set_yticklabels(device_labels)
    
    # Add "Time" label and arrow at the bottom
    arrow_y = -0.4
    ax.text(0.5, arrow_y, "Time", ha="right", va="center", fontsize=10)
    ax.annotate("", xy=(2, arrow_y), xytext=(1, arrow_y), 
                arrowprops=dict(arrowstyle="->", lw=1))
    
    # Remove the x-axis ticks
    ax.set_xticks([])
    
    # Remove the outer frame/border
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Add a legend - using 3 parts like in the reference image
    forward_patch = Rectangle((0, 0), 1, 1, facecolor=forward_color)
    backward_patch = Rectangle((0, 0), 1, 1, facecolor=backward_color)
    optimizer_patch = Rectangle((0, 0), 1, 1, facecolor=optimizer_color)
    
    legend = ax.legend(
        [forward_patch, backward_patch, optimizer_patch],
        ["Forward", "Backward", "Optimizer step"],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        ncol=3,
        frameon=False,
    )

    # Turn off grid
    ax.grid(False)

    # Save the figure
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Visualization saved to {output_file}")
