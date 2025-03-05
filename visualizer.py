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
                 - 'type': 'forward' or 'backward'
                 - 'batch': batch number
                 - 'start_time': start time of the task
                 - 'duration': duration of the task
        schedule_type: Type of scheduling algorithm used ("simple" or "1f1b")
        output_file: Path to save the visualization
    """
    # Colors for forward and backward passes
    forward_color = "royalblue"
    backward_color = "lightgreen"
    empty_color = "lightgray"

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
    fig, ax = plt.subplots(figsize=(15, 5))

    # Plot the schedule
    for device_idx, device in enumerate(schedule):
        for task in schedule[device]:
            color = forward_color if task["type"] == "forward" else backward_color
            rect = Rectangle(
                (task["start_time"], device_idx),
                task["duration"],
                0.8,
                edgecolor="black",
                facecolor=color,
                alpha=0.8,
            )
            ax.add_patch(rect)

            # Add text (batch number)
            ax.text(
                task["start_time"] + task["duration"] / 2,
                device_idx + 0.4,
                str(task["batch"]),
                ha="center",
                va="center",
                fontsize=10,
                fontweight="bold",
                color="white" if task["type"] == "forward" else "black",
            )

    # Set axis limits and labels
    ax.set_xlim(0, max_time * 1.05)
    ax.set_ylim(-0.2, num_stages + 0.2)
    ax.set_yticks(np.arange(num_stages) + 0.4)
    ax.set_yticklabels([f"Device {i+1}" for i in range(num_stages)])
    ax.set_xlabel("Time")
    ax.set_title(f"Pipeline Parallelism Schedule ({schedule_type})")

    # Add a legend
    forward_patch = Rectangle((0, 0), 1, 1, facecolor=forward_color)
    backward_patch = Rectangle((0, 0), 1, 1, facecolor=backward_color)
    ax.legend(
        [forward_patch, backward_patch],
        ["Forward Pass", "Backward Pass"],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        ncol=2,
    )

    # Add grid
    ax.grid(True, linestyle="--", alpha=0.7)

    # Save the figure
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Visualization saved to {output_file}")
