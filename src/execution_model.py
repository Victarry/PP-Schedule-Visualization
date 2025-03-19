from collections import defaultdict
from typing import Dict, List, Optional, Union


class Operation:
    """Operation is a single operation in the pipeline."""

    def __init__(self, batch_id: int, stage_id: int, op_type: str):
        self.batch_id = batch_id
        self.stage_id = stage_id
        self.op_type = op_type
        self.device_id = None

        self.start_time = None
        self.end_time = None
    
    def set_end_time(self, end_time: float):
        self.end_time = end_time
    
    def set_start_time(self, start_time: float):
        self.start_time = start_time
    
    def __repr__(self) -> str:
        return f"Operation(batch_id={self.batch_id}, stage_id={self.stage_id}, op_type={self.op_type})"

class OverlappedOperation:
    """Represents multiple operations that are overlapped/executed concurrently."""

    def __init__(self, operations: List[Operation]):
        self.operations = operations
        self.device_id = operations[0].device_id
        
        # Validate all operations are on the same device
        for op in operations:
            assert op.device_id == self.device_id, "All operations must be on the same device"
        
        # Create a combined op_type (e.g., "overlapped_forward_backward")
        self.op_type = "overlapped_" + "_".join([op.op_type for op in operations])
        
        # Use the batch_id and stage_id of the first operation for identification
        # (though we'll track all operations internally)
        self.batch_id = operations[0].batch_id
        self.stage_id = operations[0].stage_id
        
        # Initialize timing information
        self.start_time = None
        self.end_time = None
    
    def set_end_time(self, end_time: float):
        self.end_time = end_time
        for op in self.operations:
            op.set_end_time(end_time)
    
    def set_start_time(self, start_time: float):
        self.start_time = start_time
        for op in self.operations:
            op.set_start_time(start_time)

    def __repr__(self) -> str:
        op_str = ", ".join([f"({op.batch_id},{op.stage_id},{op.op_type})" for op in self.operations])
        return f"OverlappedOperation([{op_str}])"

class DeviceQueue:
    def __init__(self, stages: List[int], device_id: int):
        self.stages = stages
        self.device_id = device_id
        self.ops = []  # List of operations

    def add_operation(self, op: Operation):
        assert op.stage_id in self.stages
        self.ops.append(op)
        assert op.device_id is None
        op.device_id = self.device_id


class ScheduleConfig:
    def __init__(
        self,
        num_devices: int,
        num_stages: int,
        num_batches: int,
        p2p_latency: float = 0.0,
        placement_strategy: str = "standard",
        split_backward: bool = False,
        op_times: Optional[Dict[str, Union[float, Dict[int, float]]]] = None,
    ):
        self.num_devices = num_devices
        self.num_stages = num_stages
        self.num_batches = num_batches
        self.p2p_latency = p2p_latency
        self.placement_strategy = placement_strategy
        self.split_backward = split_backward

        # Initialize default operation times
        if self.split_backward:
            self.op_times = {
                "forward": 1.0,
                "backward_D": 1.0,
                "backward_W": 1.0,
            }
        else:
            self.op_times = {
                "forward": 1.0,
                "backward": 2.0,
            }

        # Update with user-provided operation times
        if op_times:
            for op_type, times in op_times.items():
                if isinstance(times, dict):
                    # If a dict is provided, it maps stage_id -> time
                    if op_type not in self.op_times:
                        self.op_times[op_type] = {}
                    elif not isinstance(self.op_times[op_type], dict):
                        # Convert float to dict if needed
                        self.op_times[op_type] = {i: self.op_times[op_type] for i in range(num_stages)}
                    
                    # Update with provided stage-specific times
                    for stage_id, time in times.items():
                        if not isinstance(self.op_times[op_type], dict):
                            self.op_times[op_type] = {i: self.op_times[op_type] for i in range(num_stages)}
                        self.op_times[op_type][stage_id] = time
                else:
                    # If a float is provided, use same time for all stages
                    self.op_times[op_type] = times

        assert num_stages % num_devices == 0, "num_stages must be divisible by num_devices"
        self.num_stages_per_device = num_stages // num_devices

        self.init_device_to_stages()
        assert (
            sum(len(stages) for stages in self.device_to_stages.values()) == num_stages
        )

    def init_device_to_stages(self):
        if self.placement_strategy == "standard":
            # Evenly distributed
            stages_per_device = self.num_stages // self.num_devices
            self.device_to_stages = defaultdict(list)
            for i in range(self.num_stages):
                device_to_put = i // stages_per_device
                self.device_to_stages[device_to_put].append(i)
        elif self.placement_strategy == "interleave":
            self.device_to_stages = defaultdict(list)
            for i in range(self.num_stages):
                device_to_put = i % self.num_devices
                self.device_to_stages[device_to_put].append(i)
        else:
            raise ValueError(f"Invalid placement strategy: {self.placement_strategy}")

    def get_op_time(self, op_type: str, stage_id: int):
        # For overlapped operations, extract the original operation types
        if op_type.startswith("overlapped_"):
            if op_type in self.op_times and self.op_times[op_type][stage_id]:
                return self.op_times[op_type][stage_id]
            else:
                op_parts = op_type.split("_")[1:]
                if len(op_parts) >= 2:
                    op_type1, op_type2 = op_parts[0], op_parts[1]
                    return self.get_op_time(op_type1, stage_id) + self.get_op_time(op_type2, stage_id)

        if op_type not in self.op_times:
            raise ValueError(f"Invalid operation type: {op_type}")
        times = self.op_times[op_type]
        if isinstance(times, dict):
            # If we have stage-specific times, use those
            if stage_id not in times:
                raise ValueError(f"No time specified for operation {op_type} at stage {stage_id}")
            return times[stage_id]
        else:
            # If we have a single float, use the same value for all stages
            return times


class Schedule:
    def __init__(self, config: ScheduleConfig):
        self.ops = {}  # (batch_id, stage_id, op_type) -> Operation
        self.device_queues: List[DeviceQueue] = []
        for dev_id in range(config.num_devices):
            self.device_queues.append(DeviceQueue(config.device_to_stages[dev_id], dev_id))
        self.config = config

        self.init_operations()
        self.op_to_overlapped = {}
    
    def register_overlapped_operation(self, overlapped_op: OverlappedOperation):
        for op in overlapped_op.operations:
            self.op_to_overlapped[(op.batch_id, op.stage_id, op.op_type)] = overlapped_op
            self.ops[(op.batch_id, op.stage_id, op.op_type)] = overlapped_op

    def init_operations(self):
        op_types = ["forward", "backward"]
        if self.config.split_backward:
            op_types = ["forward", "backward_D", "backward_W"]
        for batch_id in range(self.config.num_batches):
            for stage_id in range(self.config.num_stages):
                for op_type in op_types:
                    self.ops[(batch_id, stage_id, op_type)] = Operation(
                        batch_id, stage_id, op_type
                    )

    def get_op(self, batch_id: int, stage_id: int, op_type: str):
        if (batch_id, stage_id, op_type) in self.op_to_overlapped:
            return self.op_to_overlapped[(batch_id, stage_id, op_type)]
        return self.ops[(batch_id, stage_id, op_type)]

    def get_dependencies(self, op: Operation, include_device_dependency=True):
        deps = []
        if isinstance(op, OverlappedOperation):
            for sub_op in op.operations:
                deps.extend(self.get_dependencies(sub_op, include_device_dependency=False))

            if include_device_dependency:
                device_index = self.device_queues[op.device_id].ops.index(op)
                if device_index > 0:
                    deps.append((self.device_queues[op.device_id].ops[device_index - 1], 0.0))
            return deps
        if op.op_type == "forward":
            if op.stage_id > 0:
                deps.append(
                    (
                        self.get_op(op.batch_id, op.stage_id - 1, "forward"),
                        self.config.p2p_latency,
                    )
                )
        if self.config.split_backward:
            if op.op_type == "backward_D":
                if op.stage_id < self.config.num_stages - 1:
                    deps.append(
                        (
                            self.get_op(op.batch_id, op.stage_id + 1, "backward_D"),
                            self.config.p2p_latency,
                        )
                    )
            elif op.op_type == "backward_W":
                if op.stage_id < self.config.num_stages - 1:
                    deps.append(
                        (
                            self.get_op(op.batch_id, op.stage_id, "backward_D"),
                            self.config.p2p_latency,
                        )
                    )
        else:
            if op.op_type == "backward":
                if op.stage_id < self.config.num_stages - 1:
                    deps.append(
                        (
                            self.get_op(op.batch_id, op.stage_id + 1, "backward"),
                            self.config.p2p_latency,
                        )
                    )

        if include_device_dependency:
            device_index = self.device_queues[op.device_id].ops.index(op)
            if device_index > 0:
                deps.append((self.device_queues[op.device_id].ops[device_index - 1], 0.0))
        return deps

    def show(self):
        """Display detailed information about the schedule for debugging purposes."""
        print("\n=== SCHEDULE DETAILS ===")
        print(f"Devices: {self.config.num_devices}, Stages: {self.config.num_stages}, Batches: {self.config.num_batches}")
        print(f"Placement Strategy: {self.config.placement_strategy}")
        print("\n=== DEVICE QUEUES ===")
        
        for dev_id in range(self.config.num_devices):
            print(f"\nDEVICE {dev_id} (Stages: {self.device_queues[dev_id].stages}):")
            print("-" * 80)
            print(f"{'Batch':^6} | {'Stage':^6} | {'Type':^10} | {'Start':^10} | {'End':^10} | {'Duration':^10}")
            print("-" * 80)
            
            for op in self.device_queues[dev_id].ops:
                op_type = op.op_type
                start = f"{op.start_time:.2f}" if op.start_time is not None else "N/A"
                end = f"{op.end_time:.2f}" if op.end_time is not None else "N/A"
                
                duration = "N/A"
                if op.start_time is not None and op.end_time is not None:
                    duration = f"{op.end_time - op.start_time:.2f}"
                
                print(f"{op.batch_id:^6} | {op.stage_id:^6} | {op_type:^10} | {start:^10} | {end:^10} | {duration:^10}")

        # Find the total execution time (if timing info is available)
        if all(op.end_time is not None for op in self.ops.values()):
            total_time = max(op.end_time for op in self.ops.values())
            print(f"\nTotal execution time: {total_time:.2f}")
    
    def execute(self):
        # TODO: change the execution order to topological order via DAG
        def execute_op(op: Operation):
            if op.end_time is not None:
                return
            deps = self.get_dependencies(op)
            if len(deps) == 0:
                op.set_start_time(0.0)
            else:
                for dep, gap in deps:
                    if dep.end_time is None or dep.start_time is None:
                        execute_op(dep)
                op.set_start_time(max(dep.end_time + gap for dep, gap in deps))
            op.set_end_time(op.start_time + self.config.get_op_time(
                op.op_type, op.stage_id
            ))

        op_num = len(self.device_queues[0].ops)
        for i in range(op_num):
            for dev_id in range(self.config.num_devices):
                if len(self.device_queues[dev_id].ops) <= i:
                    continue
                op = self.device_queues[dev_id].ops[i]
                execute_op(op)

        for op in self.ops.values():
            assert (
                op.start_time is not None
            ), f"op {op.batch_id}, {op.stage_id}, {op.op_type} has no start time"
            assert (
                op.end_time is not None
            ), f"op {op.batch_id}, {op.stage_id}, {op.op_type} has no end time"

    def get_total_execution_time(self):
        return max(op.end_time for op in self.ops.values())
    
    def get_bubble_rate(self):
        actual_time = self.get_total_execution_time()
        ideal_time = 0
        for stage_id in range(self.config.num_stages):
            for op_type in ["forward", "backward"]:
                ideal_time += self.config.get_op_time(op_type, stage_id)
        ideal_time = ideal_time * self.config.num_batches / self.config.num_devices

        return (actual_time - ideal_time) / ideal_time

    def get_device_running_time(self):
        device_time = [0] * self.config.num_devices
        for dev_id in range(self.config.num_devices):
            for op in self.device_queues[dev_id].ops:
                device_time[dev_id] += op.end_time - op.start_time
        return device_time
