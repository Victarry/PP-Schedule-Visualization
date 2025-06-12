import json
import os
import argparse
import re
from collections import defaultdict
from src.execution_model import Schedule, ScheduleConfig, Operation
from src.visualizer import visualize_pipeline_parallelism_dash


def is_valid_event_filename(filename, pp_size, vpp_size):
    """
    Check if filename matches the expected format:
    event_times_PP{pp_size}_VPP{vpp_size}_TPxCPxDP_rank_{rank}_pp_rank_{pp_rank}_rank_{final_rank}.json

    Returns True if valid, False otherwise.
    """
    # Create regex pattern for the expected format
    pattern = rf"^event_times_PP{pp_size}_VPP{vpp_size}_TPxCPxDP_rank_\d+_pp_rank_\d+_rank_\d+\.json$"
    return bool(re.match(pattern, filename))


def parse_event_filename(filename):
    """
    Parse the event filename and extract rank information.

    Expected format: event_times_PP{pp_size}_VPP{vpp_size}_TPxCPxDP_rank_{rank}_pp_rank_{pp_rank}_rank_{final_rank}.json

    Returns: (TPxCPxDP_rank, pp_rank, global_rank) or None if parsing fails
    """
    try:
        # Remove .json extension
        name_without_ext = filename.replace(".json", "")
        parts = name_without_ext.split("_")

        # Find the TPxCPxDP part and the rank values
        tpxcpxdp_rank = None
        pp_rank = None
        global_rank = None

        for i, part in enumerate(parts):
            # Look for TPxCPxDP pattern followed by 'rank'
            if part.startswith("TP") and "CP" in part and part.endswith("DP"):
                if i + 2 < len(parts) and parts[i + 1] == "rank":
                    tpxcpxdp_rank = int(parts[i + 2])
            # Look for 'pp_rank' pattern
            elif part == "pp" and i + 2 < len(parts) and parts[i + 1] == "rank":
                pp_rank = int(parts[i + 2])
            # Look for the final 'rank' (global rank) - this should be the last rank in the filename
            elif part == "rank" and i + 1 < len(parts) and i == len(parts) - 2:
                global_rank = int(parts[i + 1])

        if tpxcpxdp_rank is None or pp_rank is None or global_rank is None:
            return None

        return (tpxcpxdp_rank, pp_rank, global_rank)

    except (ValueError, IndexError):
        return None


def load_event_times_from_json(data_dir, pp_size, vpp_size):
    """Load event times from JSON files in the specified directory."""
    all_files = [f for f in os.listdir(data_dir) if f.endswith(".json")]

    # Filter files that match the expected format
    event_files = [
        f for f in all_files if is_valid_event_filename(f, pp_size, vpp_size)
    ]

    if len(event_files) == 0:
        print(f"Available files in {data_dir}:")
        for f in all_files[:10]:  # Show first 10 files for debugging
            print(f"  {f}")
        raise ValueError(
            f"No event files found matching pattern event_times_PP{pp_size}_VPP{vpp_size}_*.json"
        )

    print(f"Found {len(event_files)} matching event files")
    event_times = {}

    for file_name in event_files:
        parsed_result = parse_event_filename(file_name)
        if parsed_result is None:
            print(f"Warning: Could not parse filename {file_name}")
            continue

        tpxcpxdp_rank, pp_rank, global_rank = parsed_result

        if tpxcpxdp_rank == 0:
            try:
                with open(os.path.join(data_dir, file_name), "r") as f:
                    event_data = json.load(f)
                    event_times[(pp_rank, tpxcpxdp_rank)] = event_data
                    print(
                        f"Loaded data from {file_name}: global_rank={global_rank}, pp_rank={pp_rank}, tpxcpxdp_rank={tpxcpxdp_rank}"
                    )
            except Exception as e:
                print(f"Error loading {file_name}: {e}")

    return event_times


def create_pp_schedule_from_event_times(event_times, pp_size):
    """Create a Schedule object from event times data."""
    # Determine number of devices/stages from the data
    num_devices = pp_size

    # Find the maximum batch ID by parsing event names
    max_batch_id = 0
    for events in event_times.values():
        for event_name in events:
            if event_name.startswith(("forward-", "backward-")):
                parts = event_name.split("-")
                if len(parts) >= 2 and parts[1].isdigit():
                    batch_id = int(parts[1])
                    max_batch_id = max(max_batch_id, batch_id)

    num_batches = max_batch_id + 1

    # Create a simple config (actual times will come from event data)
    config = ScheduleConfig(
        num_devices=num_devices,
        num_stages=num_devices,  # Assuming 1:1 mapping of devices to stages
        num_batches=num_batches,
        p2p_latency=0,  # Will be implicit in the event timing
        op_times={},  # Not needed as we'll use real timing data
        placement_strategy="standard",
    )

    # Create a schedule
    schedule = Schedule(config)

    # Populate the schedule with operations based on event times
    for (pp_rank, tpxcpxdp_rank), events in event_times.items():
        # Process forward passes
        for batch_id in range(num_batches):
            forward_start_key = f"forward-{batch_id}-start"
            forward_end_key = f"forward-{batch_id}-end"

            if forward_start_key in events and forward_end_key in events:
                # Create an operation and set its timing directly
                forward_op = Operation(batch_id, pp_rank, "forward")
                forward_op.execution_time = (
                    events[forward_end_key] - events[forward_start_key]
                )
                forward_op.start_time = events[forward_start_key]
                forward_op.end_time = events[forward_end_key]

                # Add to schedule
                schedule.ops[(batch_id, pp_rank, "forward")] = forward_op
                schedule.device_queues[pp_rank].add_operation(forward_op)

        # Process backward passes
        for batch_id in range(num_batches):
            backward_start_key = f"backward-{batch_id}-start"
            backward_end_key = f"backward-{batch_id}-end"

            if backward_start_key in events and backward_end_key in events:
                # Create an operation and set its timing directly
                backward_op = Operation(batch_id, pp_rank, "backward")
                backward_op.execution_time = (
                    events[backward_end_key] - events[backward_start_key]
                )
                backward_op.start_time = events[backward_start_key]
                backward_op.end_time = events[backward_end_key]

                # Add to schedule
                schedule.ops[(batch_id, pp_rank, "backward")] = backward_op
                schedule.device_queues[pp_rank].add_operation(backward_op)

    return schedule


def create_vpp_schedule_from_event_times(event_times, pp_size, vpp_size):
    """Create a VPP Schedule object from event times data."""
    # Determine number of devices/stages from the data
    # Find the maximum batch ID by parsing event names
    max_batch_id = 0
    for events in event_times.values():
        for event_name in events:
            if event_name.startswith(("forward-", "backward-")):
                parts = event_name.split("-")
                assert len(parts) == 4
                assert parts[0] in ["forward", "backward"]
                assert parts[1].isdigit() and parts[2].isdigit()
                assert parts[3] in ["start", "end"]
                batch_id = int(parts[2]) # backward-0-19-end
                max_batch_id = max(max_batch_id, batch_id)

    num_batches = max_batch_id + 1

    # Create a simple config (actual times will come from event data)
    config = ScheduleConfig(
        num_devices=pp_size,
        num_stages=pp_size * vpp_size,
        num_batches=num_batches,
        p2p_latency=0,  # Will be implicit in the event timing
        op_times={},  # Not needed as we'll use real timing data
        placement_strategy="interleave",
    )

    # Create a schedule
    schedule = Schedule(config)

    # Populate the schedule with operations based on event times
    for (pp_rank, tpxcpxdp_rank), events in event_times.items():
        # Process forward passes
        for model_chunk_id in range(vpp_size):
            for batch_id in range(num_batches):
                forward_start_key = f"forward-{model_chunk_id}-{batch_id}-start"
                forward_end_key = f"forward-{model_chunk_id}-{batch_id}-end"

                # Create an operation and set its timing directly
                stage_id = pp_size * model_chunk_id + pp_rank
                forward_op = Operation(batch_id, stage_id=stage_id, op_type="forward")
                forward_op.execution_time = (
                    events[forward_end_key] - events[forward_start_key]
                )
                forward_op.start_time = events[forward_start_key]
                forward_op.end_time = events[forward_end_key]

                # Add to schedule
                schedule.ops[(batch_id, stage_id, "forward")] = forward_op
                schedule.device_queues[pp_rank].add_operation(forward_op)

        # Process backward passes
        for model_chunk_id in range(vpp_size):
            for batch_id in range(num_batches):
                backward_start_key = f"backward-{model_chunk_id}-{batch_id}-start"
                backward_end_key = f"backward-{model_chunk_id}-{batch_id}-end"

                stage_id = pp_size * model_chunk_id + pp_rank
                if backward_start_key in events and backward_end_key in events:
                    # Create an operation and set its timing directly
                    backward_op = Operation(
                        batch_id, stage_id=stage_id, op_type="backward"
                    )
                    backward_op.execution_time = (
                        events[backward_end_key] - events[backward_start_key]
                    )
                    backward_op.start_time = events[backward_start_key]
                    backward_op.end_time = events[backward_end_key]

                    # Add to schedule
                    schedule.ops[(batch_id, stage_id, "backward")] = backward_op
                    schedule.device_queues[pp_rank].add_operation(backward_op)

    return schedule


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Visualize pipeline parallelism from event data"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Directory containing event_times_*.json files",
    )
    parser.add_argument(
        "--pp-size", type=int, required=True, help="Pipeline parallelism size"
    )
    parser.add_argument(
        "--vpp-size", type=int, required=True, help="Virtual pipeline parallelism size"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8050,
        help="Port for the visualization dashboard (default: 8050)",
    )
    args = parser.parse_args()

    # Load event times from JSON files
    event_times = load_event_times_from_json(args.data_dir, args.pp_size, args.vpp_size)

    # Create schedule from event times
    if args.vpp_size == 1:
        schedule = create_pp_schedule_from_event_times(event_times, args.pp_size)
    else:
        schedule = create_vpp_schedule_from_event_times(
            event_times, args.pp_size, args.vpp_size
        )

    # Calculate and print execution metrics
    total_execution_time = max(
        op.end_time for op in schedule.ops.values() if op.end_time is not None
    )
    print(f"Total execution time: {total_execution_time:.2f} ms")

    # Calculate bubble time percentage
    device_times = defaultdict(float)
    for device_id, device_queue in enumerate(schedule.device_queues):
        for op in device_queue.ops:
            if op.start_time is not None and op.end_time is not None:
                device_times[device_id] += op.end_time - op.start_time

    # Print bubble percentage for each device
    for device_id, active_time in device_times.items():
        bubble_percentage = (
            (total_execution_time - active_time) / total_execution_time * 100
        )
        print(f"Device {device_id} bubble: {bubble_percentage:.2f}%")

    # Visualize the schedule
    print("Launching visualization...")
    visualize_pipeline_parallelism_dash(
        schedule, schedule_type="1F1B-Imported", port=args.port
    )


if __name__ == "__main__":
    main()
