import matplotlib.pyplot as plt
import numpy as np
import argparse
import json
import yaml
import os
from matplotlib.patches import Rectangle
from typing import List, Tuple, Dict, Literal

# Import visualization function from the new module
from visualizer import visualize_pipeline_parallelism


def create_1f1b_schedule(
    num_stages: int,
    num_batches: int,
    forward_times: List[float],
    backward_times: List[float],
    p2p_time: float = 0.0,
) -> Dict[int, List[Dict]]:
    """
    Create a 1F1B (One-Forward-One-Backward) schedule for pipeline parallelism.

    This implementation takes a data-centric approach:
    1. First determine the operation sequence for each pipeline stage (which microbatch to process when)
    2. Then calculate timing based on dependencies between operations

    The 1F1B pattern has three phases:
    - Warmup: Forward passes for first num_stages microbatches
    - Steady state: Alternating between forward and backward passes
    - Cooldown: Backward passes for remaining microbatches

    Returns:
        A dictionary mapping device IDs to lists of tasks.
        Each task is a dictionary with keys:
        - 'type': 'forward' or 'backward'
        - 'batch': batch number
        - 'start_time': start time of the task
        - 'duration': duration of the task
    """
    # Initialize empty schedule
    schedule = {stage: [] for stage in range(num_stages)}

    # Step 1: Determine operation sequence for each stage
    # This will generate the sequence of operations (forward/backward on which microbatch)
    # that each stage should perform, without timing information yet
    operation_sequence = determine_1f1b_operation_sequence(num_stages, num_batches)

    # Step 2: Convert operation sequence to schedule with timing
    # Taking into account dependencies between operations
    schedule = calculate_operation_timing(
        operation_sequence, num_stages, forward_times, backward_times, p2p_time
    )

    return schedule


def determine_1f1b_operation_sequence(
    num_stages: int, num_batches: int
) -> Dict[int, List[Dict]]:
    """
    Determine the sequence of operations (forward/backward) for each stage in 1F1B scheduling.

    Args:
        num_stages: Number of pipeline stages
        num_batches: Number of micro-batches

    Returns:
        Dictionary mapping stage ID to a list of operations in sequence.
        Each operation is a dict with keys 'type' ('forward' or 'backward') and 'batch'.
    """
    operation_sequence = {i: [] for i in range(num_stages)}
    for current_stage in range(num_stages):
        warmup_batches = num_stages - current_stage
        for j in range(1, warmup_batches + 1):
            operation_sequence[current_stage].append({"type": "forward", "batch": j})
        steady_batches = num_batches - warmup_batches
        for j in range(warmup_batches + 1, warmup_batches + steady_batches + 1):
            operation_sequence[current_stage].append(
                {"type": "backward", "batch": j - warmup_batches}
            )
            operation_sequence[current_stage].append({"type": "forward", "batch": j})
        for j in range(warmup_batches):
            operation_sequence[current_stage].append(
                {"type": "backward", "batch": j + steady_batches + 1}
            )

    return operation_sequence


def calculate_operation_timing(
    operation_sequence: Dict[int, List[Dict]],
    num_stages: int,
    forward_times: List[float],
    backward_times: List[float],
    p2p_time: float = 0.0,
) -> Dict[int, List[Dict]]:
    """
    Recursively calculate the specific timing of each operation in a 1F1B schedule.

    When encountering an operation that depends on a previous operation that hasn't been calculated yet,
    it will recursively calculate the timing of those operations.

    Args:
        operation_sequence: Operation sequence for each stage
        num_stages: Number of pipeline stages
        forward_times: Forward propagation time for each stage
        backward_times: Backward propagation time for each stage
        p2p_time: Point-to-point communication time between stages

    Returns:
        Complete schedule with timing information, each operation includes start_time and duration
    """
    # Initialize schedule with timing information
    schedule = {i: [] for i in range(num_stages)}

    # For recording already computed operation end times
    # Format: {(stage, batch, op_type): (start_time, end_time)}
    computed_ops = {}

    # For recording the end time of the last operation for each stage
    stage_last_end_time = [0.0] * num_stages

    # Helper function: recursively calculate the time for an operation
    def compute_op_time(stage, batch, op_type):
        # Check if this operation has already been calculated
        key = (stage, batch, op_type)
        if key in computed_ops:
            return computed_ops[key]

        # Get operation duration
        duration = (
            forward_times[stage] if op_type == "forward" else backward_times[stage]
        )

        # Determine start time (dependent on other operations)
        # 1. Consider sequential dependencies on the stage (must wait for previous operation to complete)
        start_time = stage_last_end_time[stage]

        # 2. Forward pass also depends on forward pass of previous stage (if not the first stage)
        if op_type == "forward" and stage > 0:
            # Recursively calculate the time for the forward pass of the previous stage (if not calculated yet)
            prev_stage_key = (stage - 1, batch, "forward")
            if prev_stage_key not in computed_ops:
                prev_start, prev_end = compute_op_time(stage - 1, batch, "forward")
            else:
                _, prev_end = computed_ops[prev_stage_key]
            # Update start time
            start_time = max(start_time, prev_end + p2p_time)

        # 3. Backward pass depends on:
        elif op_type == "backward":
            # a. Forward pass of the same stage
            same_stage_forward_key = (stage, batch, "forward")
            if same_stage_forward_key not in computed_ops:
                _, forward_end = compute_op_time(stage, batch, "forward")
            else:
                _, forward_end = computed_ops[same_stage_forward_key]

            start_time = max(start_time, forward_end)

            # b. Backward pass of the next stage (if not the last stage)
            if stage < num_stages - 1:
                next_stage_backward_key = (stage + 1, batch, "backward")
                if next_stage_backward_key not in computed_ops:
                    _, next_backward_end = compute_op_time(stage + 1, batch, "backward")
                else:
                    _, next_backward_end = computed_ops[next_stage_backward_key]

                start_time = max(start_time, next_backward_end + p2p_time)

        # Calculate end time
        end_time = start_time + duration

        # Store calculation results
        computed_ops[key] = (start_time, end_time)

        # Update the end time of the last operation for this stage
        stage_last_end_time[stage] = end_time

        return start_time, end_time

    # Calculate time for each operation in the operation_sequence
    for i in range(len(operation_sequence[0])):
        for stage in range(num_stages):
            batch = operation_sequence[stage][i]["batch"]
            op_type = operation_sequence[stage][i]["type"]

            # Recursively calculate the time for this operation
            start_time, _ = compute_op_time(stage, batch, op_type)

            # Fill in scheduling information
            op_with_timing = operation_sequence[stage][i].copy()
            op_with_timing["start_time"] = start_time
            op_with_timing["duration"] = (
                forward_times[stage] if op_type == "forward" else backward_times[stage]
            )
            schedule[stage].append(op_with_timing)

    return schedule


def get_bubble_rate(schedule: Dict[int, List[Dict]]):
    num_stages = len(schedule)

    max_time = 0
    for device in schedule:
        for task in schedule[device]:
            end_time = task["start_time"] + task["duration"]
            if end_time > max_time:
                max_time = end_time

    total_execution_time = max_time * num_stages

    total_computation_time = 0
    device_computation_times = {}

    for device in schedule:
        device_computation_time = 0
        for task in schedule[device]:
            device_computation_time += task["duration"]
        device_computation_times[device] = device_computation_time
        total_computation_time += device_computation_time

    bubble_rate = (
        total_execution_time - total_computation_time
    ) / total_computation_time
    return bubble_rate


def read_config_file(config_path):
    """
    Read configuration from a JSON or YAML file.

    Args:
        config_path: Path to the config file (JSON or YAML)

    Returns:
        Dictionary containing configuration parameters
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    file_ext = os.path.splitext(config_path)[1].lower()

    try:
        with open(config_path, "r") as f:
            if file_ext == ".json":
                config = json.load(f)
            elif file_ext in (".yaml", ".yml"):
                config = yaml.safe_load(f)
            else:
                raise ValueError(
                    f"Unsupported config file format: {file_ext}. Use .json, .yaml, or .yml"
                )
        return config
    except Exception as e:
        raise ValueError(f"Error reading config file: {str(e)}")


def parse_args():
    """
    Parse command-line arguments for the pipeline parallelism tool.

    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(
        description="Pipeline Parallelism Scheduler and Visualizer",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Config file option
    parser.add_argument(
        "--config", "-c", type=str, help="Path to config file (JSON or YAML)"
    )

    # Main parameters
    parser.add_argument(
        "--num-stages",
        "-s",
        type=int,
        default=0,
        help="Number of pipeline stages (devices)",
    )

    parser.add_argument(
        "--num-batches", "-b", type=int, default=0, help="Number of micro-batches"
    )

    # Forward and backward times
    parser.add_argument(
        "--forward-times",
        "-f",
        type=float,
        nargs="+",
        help="Time for forward pass at each stage (space-separated list)",
    )

    parser.add_argument(
        "--backward-times",
        "-bw",
        type=float,
        nargs="+",
        help="Time for backward pass at each stage (space-separated list)",
    )

    # Output options
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="pipeline_1f1b.png",
        help="Output file path for visualization",
    )

    parser.add_argument(
        "--no-visualization", action="store_true", help="Skip visualization generation"
    )

    parser.add_argument(
        "--p2p-time",
        type=float,
        default=0.0,
        help="Time for point-to-point communication between stages",
    )

    return parser.parse_args()


def example_usage():
    """Example usage of the visualization function and testing the scheduling algorithms."""
    # Example parameters
    num_stages = 4  # Number of pipeline stages (devices)
    num_batches = 10  # Number of micro-batches

    # Example times for forward and backward passes for each stage
    forward_times = [1.0, 1.0, 1.0, 1.0]  # Time for forward pass at each stage
    backward_times = [2.0, 2.0, 2.0, 2.0]  # Time for backward pass at each stage

    # Create 1F1B schedule
    schedule = create_1f1b_schedule(
        num_stages=num_stages,
        num_batches=num_batches,
        forward_times=forward_times,
        backward_times=backward_times,
    )

    # Create visualization with the schedule
    visualize_pipeline_parallelism(
        schedule=schedule, schedule_type="1f1b", output_file="pipeline_1f1b.png"
    )

    # Analyze the schedule
    bubble_rate = get_bubble_rate(schedule)
    print(f"Bubble rate: {bubble_rate:.4f}")


def main():
    """
    Main function that parses arguments and runs the pipeline parallelism analysis.
    """
    args = parse_args()

    # Initialize with default values
    num_stages = 4
    num_batches = 10
    forward_times = None
    backward_times = None
    output_file = "pipeline_1f1b.png"
    p2p_time = 0.0

    # Command line arguments override config file
    num_stages = args.num_stages
    num_batches = args.num_batches
    forward_times = args.forward_times
    backward_times = args.backward_times
    output_file = args.output
    p2p_time = args.p2p_time
    
    # Read from config file if provided
    if args.config:
        try:
            print(f"Reading configuration from {args.config}")
            config = read_config_file(args.config)

            # Update parameters from config
            num_stages = config.get("num_stages", num_stages)
            num_batches = config.get("num_batches", num_batches)
            forward_times = config.get("forward_times")
            backward_times = config.get("backward_times")
            output_file = config.get("output_file", output_file)
            p2p_time = config.get("p2p_time", 0.0)

        except Exception as e:
            print(f"Error reading config file: {str(e)}")
            print("Falling back to command line arguments or defaults")

    # Validate inputs
    if forward_times is None:
        forward_times = [1.0] * num_stages
    elif len(forward_times) != num_stages:
        print(
            f"Warning: forward_times length ({len(forward_times)}) doesn't match num_stages ({num_stages})"
        )
        if len(forward_times) < num_stages:
            # Extend with repeats of the last value
            forward_times = list(forward_times) + [forward_times[-1]] * (
                num_stages - len(forward_times)
            )
        else:
            # Truncate
            forward_times = forward_times[:num_stages]
        print(f"Adjusted forward_times: {forward_times}")

    if backward_times is None:
        backward_times = [2.0] * num_stages
    elif len(backward_times) != num_stages:
        print(
            f"Warning: backward_times length ({len(backward_times)}) doesn't match num_stages ({num_stages})"
        )
        if len(backward_times) < num_stages:
            # Extend with repeats of the last value
            backward_times = list(backward_times) + [backward_times[-1]] * (
                num_stages - len(backward_times)
            )
        else:
            # Truncate
            backward_times = backward_times[:num_stages]
        print(f"Adjusted backward_times: {backward_times}")

    print(f"Running with parameters:")
    print(f"  num_stages: {num_stages}")
    print(f"  num_batches: {num_batches}")
    print(f"  forward_times: {forward_times}")
    print(f"  backward_times: {backward_times}")
    print(f"  output_file: {output_file}")

    # Create 1F1B schedule
    schedule = create_1f1b_schedule(
        num_stages=num_stages,
        num_batches=num_batches,
        forward_times=forward_times,
        backward_times=backward_times,
        p2p_time=p2p_time,
    )

    # Create visualization unless --no-visualization is specified
    if not args.no_visualization:
        visualize_pipeline_parallelism(
            schedule=schedule, schedule_type="1f1b", output_file=output_file
        )

    # Analyze the schedule
    bubble_rate = get_bubble_rate(schedule)
    print(f"Bubble rate: {bubble_rate:.4f}")

    return {
        "schedule": schedule,
        "bubble_rate": bubble_rate,
        "num_stages": num_stages,
        "num_batches": num_batches,
    }


if __name__ == "__main__":
    main()
