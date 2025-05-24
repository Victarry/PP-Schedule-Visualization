from collections import defaultdict, deque
from src.execution_model import OverlappedOperation, Operation, Schedule, ScheduleConfig


def generate_1f1b_schedule(config: ScheduleConfig):
    schedule = Schedule(config)

    assert (
        config.num_devices == config.num_stages
    ), "num_devices must be equal to num_stages for 1F1B"

    for i in range(config.num_devices):
        fwd_batch_id = 0
        bwd_batch_id = 0
        cooldown_batches = warmup_batches = config.num_devices - i - 1
        steady_batches = config.num_batches - warmup_batches

        for _ in range(warmup_batches):
            schedule.device_queues[i].add_operation(
                schedule.get_op(fwd_batch_id, i, "forward")
            )
            fwd_batch_id += 1

        for _ in range(steady_batches):
            schedule.device_queues[i].add_operation(
                schedule.get_op(fwd_batch_id, i, "forward")
            )
            fwd_batch_id += 1
            schedule.device_queues[i].add_operation(
                schedule.get_op(bwd_batch_id, i, "backward")
            )
            bwd_batch_id += 1

        for _ in range(cooldown_batches):
            schedule.device_queues[i].add_operation(
                schedule.get_op(bwd_batch_id, i, "backward")
            )
            bwd_batch_id += 1

    return schedule


def generate_zero_bubble_1p_schedule(config: ScheduleConfig):
    # Create a new schedule with split_backward=True to support backward_D and backward_W operations
    schedule = Schedule(config)
    total_batches = config.num_batches
    assert (
        config.num_devices == config.num_stages
    ), "num_devices must be equal to num_stages for ZB-1P"
    assert config.split_backward, "ZB-1P requires split_backward=True"

    for i in range(config.num_devices):
        fwd_batch_id = 0
        bwd_d_batch_id = 0
        bwd_w_batch_id = 0

        cooldown_batches = warmup_batches = config.num_devices - i - 1
        steady_batches = total_batches - warmup_batches

        for _ in range(warmup_batches):
            schedule.device_queues[i].add_operation(
                schedule.get_op(fwd_batch_id, i, "forward")
            )
            fwd_batch_id += 1

        for _ in range(steady_batches):
            schedule.device_queues[i].add_operation(
                schedule.get_op(fwd_batch_id, i, "forward")
            )
            schedule.device_queues[i].add_operation(
                schedule.get_op(bwd_d_batch_id, i, "backward_D")
            )
            if fwd_batch_id - bwd_w_batch_id >= config.num_devices - 1:
                schedule.device_queues[i].add_operation(
                    schedule.get_op(bwd_w_batch_id, i, "backward_W")
                )
                bwd_w_batch_id += 1
            bwd_d_batch_id += 1
            fwd_batch_id += 1

        for _ in range(cooldown_batches):
            schedule.device_queues[i].add_operation(
                schedule.get_op(bwd_d_batch_id, i, "backward_D")
            )

            schedule.device_queues[i].add_operation(
                schedule.get_op(bwd_w_batch_id, i, "backward_W")
            )

            bwd_w_batch_id += 1
            bwd_d_batch_id += 1

        while bwd_w_batch_id < total_batches:
            schedule.device_queues[i].add_operation(
                schedule.get_op(bwd_w_batch_id, i, "backward_W")
            )
            bwd_w_batch_id += 1

    return schedule


def generate_1f1b_overlap_schedule(config: ScheduleConfig):
    schedule = Schedule(config)

    assert (
        config.num_devices == config.num_stages
    ), "num_devices must be equal to num_stages for 1F1B"

    for i in range(config.num_devices):
        fwd_batch_id = 0
        bwd_batch_id = 0
        cooldown_batches = warmup_batches = 2 * (config.num_devices - i - 1) + 1
        steady_batches = config.num_batches - warmup_batches

        for _ in range(warmup_batches):
            schedule.device_queues[i].add_operation(
                schedule.get_op(fwd_batch_id, i, "forward")
            )
            fwd_batch_id += 1

        for _ in range(steady_batches):
            fwd_op = schedule.get_op(fwd_batch_id, i, "forward")
            bwd_op = schedule.get_op(bwd_batch_id, i, "backward")
            overlapped_op = OverlappedOperation([fwd_op, bwd_op])
            schedule.register_overlapped_operation(overlapped_op)
            schedule.device_queues[i].add_operation(overlapped_op)

            fwd_batch_id += 1
            bwd_batch_id += 1

        for _ in range(cooldown_batches):
            schedule.device_queues[i].add_operation(
                schedule.get_op(bwd_batch_id, i, "backward")
            )
            bwd_batch_id += 1

    return schedule


def _get_pp_rank_microbatches(
    num_microbatches,
    num_devices,
    device_id,
    num_stages_per_device,
    microbatch_group_size_per_vp_stage,
):
    """Get the number of total, warmup, and remaining microbatches in PP scheduling."""
    total_num_microbatches = num_microbatches * num_stages_per_device

    if num_devices > 1:
        # Run (num_model_chunks-1)*microbatch_group_size_per_vp_stage on
        # all workers, followed by more microbatches after depending on
        # stage ID (more forward passes for earlier stages, later stages can
        # immediately start with 1F1B).
        num_warmup_microbatches = (num_devices - device_id - 1) * 2
        num_warmup_microbatches += (
            num_stages_per_device - 1
        ) * microbatch_group_size_per_vp_stage
    else:
        # forward_backward_no_pipelining
        num_warmup_microbatches = 1

    if num_warmup_microbatches >= total_num_microbatches:
        num_warmup_microbatches = total_num_microbatches

    return num_warmup_microbatches


def _get_schedule_table(
    num_microbatches, num_model_chunks, microbatch_group_size_per_vp_stage
):
    """Get the schedule table for PP scheduling.

    Create a tunable schedule lookup table.
    The schedule lookup table uses the virtual_microbatch_id to find the corresponding microbatch_id and model_chunk_id.
    For example, the tunable schedule table for PP2 N3M5 with VP2 is constructed as below:
    virtual_microbatch_id | 0 1 2 3 4 5 6 7 8 9
    microbatch_id         | 0 1 2 0 1 2 3 4 3 4
    model_chunk_id        | 0 0 0 1 1 1 0 0 1 1
    """
    schedule_table = []
    for min_microbatch_id_in_group in range(
        0, num_microbatches, microbatch_group_size_per_vp_stage
    ):
        if (
            min_microbatch_id_in_group + microbatch_group_size_per_vp_stage
            >= num_microbatches
        ):
            # Construct schedule for the last microbatch group
            schedule_table.extend(
                [
                    (microbatch_id, model_chunk_id)
                    for model_chunk_id in range(num_model_chunks)
                    for microbatch_id in range(
                        min_microbatch_id_in_group, num_microbatches
                    )
                ]
            )
        else:
            # Construct schedule for other microbatch groups
            schedule_table.extend(
                [
                    (microbatch_id, model_chunk_id)
                    for model_chunk_id in range(num_model_chunks)
                    for microbatch_id in range(
                        min_microbatch_id_in_group,
                        min_microbatch_id_in_group + microbatch_group_size_per_vp_stage,
                    )
                ]
            )
    return schedule_table


def _convert_schedule_table_to_order(
    num_warmup_microbatches, num_model_chunks, schedule_table
):
    """Convert a tunable schedule lookup table to the te.make_graphed_callables() accepted
    order format. For example, the tunable schedule table for PP2 N3M5 with VP2 is as below:
    virtual_microbatch_id | 0 1 2 3 4 5 6 7 8 9
    microbatch_id         | 0 1 2 0 1 2 3 4 3 4
    model_chunk_id        | 0 0 0 1 1 1 0 0 1 1

    Then the forward backward separated order is:
    forward               | 1 1 1 2 2 2 1 1 2 2
    backward              | -2 -2 -2 -1 -1 -1 -2 -2 -1 -1

    If num_warmup_microbatches is 5, the output order is:
    1 1 1 2 2 2 -2 1 -2 1 -2 2 -1 2 -1 -1 -2 -2 -1 -1
    """
    _, model_chunk_id_table = zip(*schedule_table)
    forward_order = [chunk_id + 1 for chunk_id in model_chunk_id_table]
    backward_order = [chunk_id - num_model_chunks for chunk_id in model_chunk_id_table]
    order = forward_order[:num_warmup_microbatches]
    for i in range(num_warmup_microbatches, len(forward_order)):
        order.append(forward_order[i])
        order.append(backward_order[i - num_warmup_microbatches])
    if num_warmup_microbatches > 0:
        order.extend(backward_order[-num_warmup_microbatches:])
    return order


# Some codes are copied from Megatron-LM
def generate_1f1b_interleave_schedule(config: ScheduleConfig):
    schedule = Schedule(config)

    for device_id in range(config.num_devices):
        microbatch_group_size_per_vp_stage = config.microbatch_group_size_per_vp_stage
        num_warmup_microbatches = _get_pp_rank_microbatches(
            config.num_batches,
            config.num_devices,
            device_id,
            config.num_stages_per_device,
            microbatch_group_size_per_vp_stage,
        )

        schedule_table = _get_schedule_table(
            config.num_batches,
            config.num_stages_per_device,
            microbatch_group_size_per_vp_stage,
        )

        order = _convert_schedule_table_to_order(
            num_warmup_microbatches,
            num_model_chunks=config.num_stages_per_device,
            schedule_table=schedule_table,
        )

        cur_stage_microbatch_id = {}
        for i in range(1, config.num_stages_per_device + 1):
            cur_stage_microbatch_id[i] = 0
            cur_stage_microbatch_id[-i] = 0
        for order_item in order:
            stage_id = schedule.device_queues[device_id].stages[abs(order_item) - 1]

            if order_item > 0:
                op_type = "forward"
                micro_batch_id = cur_stage_microbatch_id[order_item]
                cur_stage_microbatch_id[order_item] = (
                    cur_stage_microbatch_id[order_item] + 1
                )
            elif order_item < 0:
                op_type = "backward"
                micro_batch_id = cur_stage_microbatch_id[order_item]
                cur_stage_microbatch_id[order_item] = (
                    cur_stage_microbatch_id[order_item] + 1
                )
            else:
                raise ValueError(f"Invalid order item: {order_item}")
            schedule.device_queues[device_id].add_operation(
                schedule.get_op(micro_batch_id, stage_id, op_type)
            )
    return schedule


def generate_1f1b_interleave_overlap_schedule(config: ScheduleConfig):
    schedule = Schedule(config)

    for device_id in range(config.num_devices):
        microbatch_group_size_per_vp_stage = config.num_devices
        num_warmup_microbatches = _get_pp_rank_microbatches(
            config.num_batches,
            config.num_devices,
            device_id,
            config.num_stages_per_device,
            microbatch_group_size_per_vp_stage,
        )

        schedule_table = _get_schedule_table(
            config.num_batches,
            config.num_stages_per_device,
            microbatch_group_size_per_vp_stage,
        )

        # NOTE: Add one more warmup microbatch for overlapped operations!
        num_warmup_microbatches += 1
        order = _convert_schedule_table_to_order(
            num_warmup_microbatches,
            num_model_chunks=config.num_stages_per_device,
            schedule_table=schedule_table,
        )

        cur_stage_microbatch_id = {}
        for i in range(1, config.num_stages_per_device + 1):
            cur_stage_microbatch_id[i] = 0
            cur_stage_microbatch_id[-i] = 0
        i = 0

        num_overlapped_batches = len(order) - num_warmup_microbatches * 2
        while i < len(order):
            if i < num_warmup_microbatches:
                order_item = order[i]
                assert order_item > 0
                op_type = "forward"
                micro_batch_id = cur_stage_microbatch_id[order_item]
                cur_stage_microbatch_id[order_item] = (
                    cur_stage_microbatch_id[order_item] + 1
                )

                stage_id = schedule.device_queues[device_id].stages[abs(order_item) - 1]
                schedule.device_queues[device_id].add_operation(
                    schedule.get_op(micro_batch_id, stage_id, op_type)
                )
                i += 1
            elif (
                i >= num_warmup_microbatches
                and i < num_warmup_microbatches + num_overlapped_batches - 1
            ):
                order_item_a = order[i]
                order_item_b = order[i + 1]

                op_type_a = "forward" if order_item_a > 0 else "backward"
                micro_batch_id_a = cur_stage_microbatch_id[order_item_a]
                cur_stage_microbatch_id[order_item_a] = (
                    cur_stage_microbatch_id[order_item_a] + 1
                )

                op_type_b = "forward" if order_item_b > 0 else "backward"
                micro_batch_id_b = cur_stage_microbatch_id[order_item_b]
                cur_stage_microbatch_id[order_item_b] = (
                    cur_stage_microbatch_id[order_item_b] + 1
                )

                stage_id_a = schedule.device_queues[device_id].stages[
                    abs(order_item_a) - 1
                ]
                stage_id_b = schedule.device_queues[device_id].stages[
                    abs(order_item_b) - 1
                ]

                op_a = schedule.get_op(micro_batch_id_a, stage_id_a, op_type_a)
                op_b = schedule.get_op(micro_batch_id_b, stage_id_b, op_type_b)
                overlapped_op = OverlappedOperation([op_a, op_b])
                schedule.register_overlapped_operation(overlapped_op)
                schedule.device_queues[device_id].add_operation(overlapped_op)

                i += 2
            else:
                assert i >= num_warmup_microbatches + num_overlapped_batches
                order_item = order[i]
                assert order_item < 0
                op_type = "backward"
                micro_batch_id = cur_stage_microbatch_id[order_item]
                cur_stage_microbatch_id[order_item] = (
                    cur_stage_microbatch_id[order_item] + 1
                )

                stage_id = schedule.device_queues[device_id].stages[abs(order_item) - 1]
                schedule.device_queues[device_id].add_operation(
                    schedule.get_op(micro_batch_id, stage_id, op_type)
                )
                i += 1

    return schedule


def create_overlapped_ops(schedule, batch_id1, batch_id2, stage_id, type1, type2):
    """
    Helper function to create overlapped operations correctly.
    This handles the underlying operation creation and registration to avoid device_id issues.
    """
    # Get the operations from the schedule
    op1 = schedule.ops[(batch_id1, stage_id, type1)]
    op2 = schedule.ops[(batch_id2, stage_id, type2)]

    # Create the overlapped operation
    overlapped_op = OverlappedOperation([op1, op2])

    # Register in the schedule to ensure proper tracking
    schedule.register_overlapped_operation(overlapped_op)

    return overlapped_op


def generate_dualpipe_schedule(config: ScheduleConfig):
    """
    Implements the DualPipe scheduling strategy.

    DualPipe is a bidirectional pipeline parallelism algorithm that achieves full overlap of forward
    and backward computation-communication phases and reduces pipeline bubbles.

    The DualPipe strategy has the following characteristics:
    1. Requires placement_strategy="dualpipe" in ScheduleConfig (set automatically)
    2. Each device handles both a forward stage and a reverse stage
    3. Overlaps forward and backward operations to reduce bubble size
    4. Assumes config.num_batches corresponds to half the total microbatches in original paper (M).
    5. Currently only supports split_backward=True.

    Args:
        config: The scheduling configuration

    Returns:
        A Schedule object with the DualPipe scheduling
    """
    # Ensure placement strategy is set for Schedule initialization
    assert (
        config.placement_strategy == "dualpipe"
    ), "DualPipe schedule currently only supports placement_strategy='dualpipe'"
    # Assertions based on DualPipe requirements
    assert (
        config.num_stages % 2 == 0
    ), "DualPipe requires an even number of stages (and devices)"
    assert (
        config.num_devices == config.num_stages
    ), "DualPipe requires num_devices == num_stages"
    assert (
        config.num_batches % 2 == 0
    ), "DualPipe requires an even number of microbatches (config.num_batches)"
    # Assertion based on original implementation: num_chunks >= num_ranks * 2
    # Here, M (config.num_batches) corresponds to half_num_chunks
    assert (
        config.num_batches >= config.num_devices
    ), "DualPipe requires config.num_batches >= config.num_devices"
    assert (
        config.split_backward
    ), "DualPipe schedule currently only supports split_backward=True"

    schedule = Schedule(config, init_ops=False)

    num_stages = config.num_stages
    num_devices = config.num_devices
    # config.num_batches is M in the original paper, which corresponds to half_num_chunks
    half_num_chunks = config.num_batches // 2
    num_half_ranks = num_devices // 2

    fwd_batch_ids = defaultdict(int)  # (device_id, phase) -> batch_id
    bwd_d_batch_ids = defaultdict(int)  # (device_id, phase) -> batch_id

    waited_weight_grad = [
        deque() for _ in range(num_devices)
    ]  # (device_id, ) -> List[(stage_id, batch_id)]

    for device_id in range(num_devices):
        is_in_second_half = device_id >= num_half_ranks
        if is_in_second_half:
            fwd_batch_ids[device_id, 1] = 0
            fwd_batch_ids[device_id, 0] = config.num_batches // 2
            bwd_d_batch_ids[device_id, 1] = 0
            bwd_d_batch_ids[device_id, 0] = config.num_batches // 2
        else:
            fwd_batch_ids[device_id, 0] = 0
            fwd_batch_ids[device_id, 1] = config.num_batches // 2
            bwd_d_batch_ids[device_id, 0] = 0
            bwd_d_batch_ids[device_id, 1] = config.num_batches // 2

    def get_stage_for_phase(device_id, phase, num_stages, is_in_second_half):
        stage_fwd_dir = device_id  # Stage handled when moving forward (0 to N-1)
        stage_rev_dir = (
            num_stages - 1 - device_id
        )  # Stage handled when moving backward (N-1 to 0)
        if not is_in_second_half:
            # First half: phase 0 -> fwd_dir, phase 1 -> rev_dir
            return stage_fwd_dir if phase == 0 else stage_rev_dir
        else:
            # Second half: phase 0 -> rev_dir, phase 1 -> fwd_dir
            return stage_rev_dir if phase == 0 else stage_fwd_dir

    def add_op_to_queue(device_id, stage_id, op_type, batch_id):
        # Retrieve the correct pre-initialized Operation object
        op = Operation(batch_id, stage_id, op_type)
        schedule.register_operation(op)
        # Add to the device queue
        schedule.device_queues[device_id].add_operation(op)

    def _schedule_forward_chunk(device_id, phase, is_in_second_half):
        """Schedules a forward compute operation."""
        stage_id = get_stage_for_phase(device_id, phase, num_stages, is_in_second_half)
        batch_id = fwd_batch_ids[device_id, phase]
        add_op_to_queue(device_id, stage_id, "forward", batch_id)
        fwd_batch_ids[device_id, phase] += 1

    def _schedule_backward_chunk(device_id, phase, is_in_second_half):
        """Schedules a backward_D with backward_W compute operation."""
        stage_id = get_stage_for_phase(device_id, phase, num_stages, is_in_second_half)
        batch_id = bwd_d_batch_ids[device_id, phase]
        add_op_to_queue(device_id, stage_id, "backward", batch_id)
        bwd_d_batch_ids[device_id, phase] += 1

    def _schedule_backward_input_chunk(device_id, phase, is_in_second_half):
        """Schedules a backward_D compute operation."""
        stage_id = get_stage_for_phase(device_id, phase, num_stages, is_in_second_half)
        batch_id = bwd_d_batch_ids[device_id, phase]
        add_op_to_queue(device_id, stage_id, "backward_D", batch_id)
        bwd_d_batch_ids[device_id, phase] += 1
        waited_weight_grad[device_id].append((stage_id, batch_id))

    def _schedule_backward_weight_chunk(device_id):
        """Schedules a backward_W compute operation."""
        stage_id, batch_id = waited_weight_grad[device_id].popleft()
        add_op_to_queue(device_id, stage_id, "backward_W", batch_id)

    def _schedule_forward_backward_chunk(
        device_id, fwd_phase, bwd_phase, is_in_second_half
    ):
        """Schedules an overlapped forward and backward_D compute operation."""
        fwd_stage_id = get_stage_for_phase(
            device_id, fwd_phase, num_stages, is_in_second_half
        )
        bwd_stage_id = get_stage_for_phase(
            device_id, bwd_phase, num_stages, is_in_second_half
        )

        fwd_batch_id = fwd_batch_ids[device_id, fwd_phase]

        fwd_op = Operation(fwd_batch_id, fwd_stage_id, "forward")
        schedule.register_operation(fwd_op)
        fwd_batch_ids[device_id, fwd_phase] += 1

        bwd_batch_id_d = bwd_d_batch_ids[device_id, bwd_phase]
        bwd_op = Operation(bwd_batch_id_d, bwd_stage_id, "backward")
        schedule.register_operation(bwd_op)
        bwd_d_batch_ids[device_id, bwd_phase] += 1

        # Create and register the overlapped operation
        overlapped_op = OverlappedOperation([fwd_op, bwd_op])
        schedule.register_overlapped_operation(overlapped_op)

        # Add the overlapped operation to the queue
        schedule.device_queues[device_id].add_operation(overlapped_op)

    # Process each device (rank in original code)
    for device_id in range(num_devices):
        half_rank = min(device_id, num_devices - 1 - device_id)
        is_in_second_half = device_id >= num_half_ranks
        is_middle_rank = (device_id == num_half_ranks - 1) or (
            device_id == num_half_ranks
        )

        # Map original steps to operation additions
        # Step 1: nF0
        step_1_count = (num_half_ranks - half_rank - 1) * 2
        for _ in range(step_1_count):
            _schedule_forward_chunk(device_id, 0, is_in_second_half)  # F0

        # Step 2: nF0F1
        step_2_count = half_rank + 1
        for i in range(step_2_count):
            _schedule_forward_chunk(device_id, 0, is_in_second_half)  # F0
            _schedule_forward_chunk(device_id, 1, is_in_second_half)  # F1

        # Step 3: nB1W1F1
        step_3_count = num_half_ranks - half_rank - 1
        for _ in range(step_3_count):
            _schedule_backward_input_chunk(device_id, 1, is_in_second_half)  # B1_D
            _schedule_backward_weight_chunk(
                device_id,
            )  # W1
            _schedule_forward_chunk(device_id, 1, is_in_second_half)  # F1

        # Step 4 (Main step): nF0B1F1B0
        step_4_count = half_num_chunks - num_devices + half_rank + 1
        for i in range(step_4_count):
            if i == 0:
                if is_middle_rank:
                    # Schedule F0, B1_D, W1 sequentially for middle ranks on first iteration
                    _schedule_forward_chunk(device_id, 0, is_in_second_half) # F0
                    _schedule_backward_chunk(device_id, 1, is_in_second_half)# B1
                else:
                    # Overlap F0 and B1_D, then schedule W1
                    _schedule_forward_backward_chunk(
                        device_id, 0, 1, is_in_second_half
                    )  # F0+B1
            else:
                _schedule_forward_backward_chunk(
                    device_id, 0, 1, is_in_second_half
                )  # F0+B1
            # Overlap F1 and B0_D, then schedule W0
            _schedule_forward_backward_chunk(
                device_id, 1, 0, is_in_second_half
            )  # F1+B0

        # Step 5: nB1F1B0
        step_5_count = num_half_ranks - half_rank - 1
        for _ in range(step_5_count):
            _schedule_backward_chunk(device_id, 1, is_in_second_half)  # B1_D + B1_W
            _schedule_forward_backward_chunk(
                device_id, 1, 0, is_in_second_half
            )  # F1+B0

        # Step 6: nB1B0
        step_6_count = half_rank + 1
        enable_zb = False
        for i in range(step_6_count):
            if i == step_6_count // 2 and half_rank % 2 == 1:
                enable_zb = True
            if enable_zb:
                _schedule_backward_input_chunk(device_id, 1, is_in_second_half)
            else:
                _schedule_backward_chunk(device_id, 1, is_in_second_half)
            if i == step_6_count // 2 and half_rank % 2 == 0:
                enable_zb = True
            if enable_zb:
                _schedule_backward_input_chunk(device_id, 0, is_in_second_half)
            else:
                _schedule_backward_chunk(device_id, 0, is_in_second_half)

        # Step 7: nWB0
        step_7_count = num_half_ranks - half_rank - 1
        for _ in range(step_7_count):
            _schedule_backward_weight_chunk(
                device_id
            )  # W1 (use gradient from B1_D scheduled previously)
            _schedule_backward_input_chunk(device_id, 0, is_in_second_half)  # B0_D

        # Step 8: nW
        step_8_count = half_rank + 1
        for _ in range(step_8_count):
            # W0 uses gradients from B0_D scheduled in steps 4, 5, 6.
            # W1 uses gradients from B1_D scheduled in steps 3, 4, 5, 6.
            # The last W0 gradients correspond to B0_D from step 6 or 7.
            _schedule_backward_weight_chunk(
                device_id
            )  # W0 (use gradient from B0_D scheduled previously)

    return schedule


def generate_dualpipe_v_schedule(config: ScheduleConfig):
    """
    Implements the DualPipe-V scheduling strategy based on dualpipe_v.py.

    DualPipe-V aims to improve upon DualPipe by utilizing Zero Bubble (ZB)
    techniques, further reducing pipeline bubbles by overlapping gradient
    computation (backward_D) and weight updates (backward_W).

    Key characteristics:
    1. Requires placement_strategy="dualpipe".
    2. Each device handles a forward stage and a reverse stage.
    3. Requires split_backward=True.
    4. Overlaps forward (F) and backward_D (B_D) operations.
    5. Schedules backward_W (W) operations separately.
    6. Uses Zero Bubble logic in later steps to delay W operations.
    7. Assumes config.num_batches corresponds to the total number of microbatches (`num_chunks` in dualpipe_v.py).

    Args:
        config: The scheduling configuration.

    Returns:
        A Schedule object with the DualPipe-V scheduling.
    """
    schedule = Schedule(config, init_ops=False)

    assert config.num_stages == config.num_devices * 2, "num_stages must be equal to num_devices * 2 for DualPipe-V"
    assert config.split_backward, "DualPipe-V requires split_backward=True"

    num_stages = config.num_stages
    num_devices = config.num_devices

    fwd_batch_ids = defaultdict(int)  # (device_id, chunk_id) -> batch_id
    bwd_d_batch_ids = defaultdict(int)  # (device_id, chunk_id) -> batch_id

    waited_weight_grad = [
        deque() for _ in range(num_devices)
    ]  # (device_id, ) -> List[(stage_id, batch_id)]

    for device_id in range(num_devices):
        fwd_batch_ids[device_id, 0] = 0
        fwd_batch_ids[device_id, 1] = 0
        bwd_d_batch_ids[device_id, 0] = 0
        bwd_d_batch_ids[device_id, 1] = 0


    def add_op_to_queue(device_id, stage_id, op_type, batch_id):
        # Retrieve the correct pre-initialized Operation object
        op = Operation(batch_id, stage_id, op_type)
        schedule.register_operation(op)
        # Add to the device queue
        schedule.device_queues[device_id].add_operation(op)
    
    def get_stage_for_chunk(device_id, chunk_id):
        if chunk_id == 0:
            # Forward direction stage for this device
            return device_id
        else:
            # Reverse direction stage for this device
            return num_stages - 1 - device_id

    def _schedule_forward_chunk(device_id, chunk_id):
        """Schedules a forward compute operation."""
        stage_id = get_stage_for_chunk(device_id, chunk_id)
        batch_id = fwd_batch_ids[device_id, chunk_id]
        add_op_to_queue(device_id, stage_id, "forward", batch_id)
        fwd_batch_ids[device_id, chunk_id] += 1

    def _schedule_backward_chunk(device_id, chunk_id, enable_zb=False):
        """Schedules a backward_D compute operation."""
        stage_id = get_stage_for_chunk(device_id, chunk_id)
        batch_id = bwd_d_batch_ids[device_id, chunk_id]
        if enable_zb:
            add_op_to_queue(device_id, stage_id, "backward_D", batch_id)
            waited_weight_grad[device_id].append((stage_id, batch_id))
        else:
            add_op_to_queue(device_id, stage_id, "backward", batch_id)
        bwd_d_batch_ids[device_id, chunk_id] += 1

    def _schedule_backward_weight_chunk(device_id):
        """Schedules a backward_W compute operation."""
        assert waited_weight_grad[device_id], f"Device {device_id} has no waited weight grads to schedule"
        stage_id, batch_id = waited_weight_grad[device_id].popleft()
        add_op_to_queue(device_id, stage_id, "backward_W", batch_id)

    def _schedule_forward_backward_chunk(
        device_id, fwd_chunk_id, bwd_chunk_id
    ):
        """Schedules an overlapped forward and backward_D compute operation."""
        fwd_stage_id = get_stage_for_chunk(device_id, fwd_chunk_id)
        bwd_stage_id = get_stage_for_chunk(device_id, bwd_chunk_id)

        fwd_batch_id = fwd_batch_ids[device_id, fwd_chunk_id]
        fwd_op = Operation(fwd_batch_id, fwd_stage_id, "forward")
        schedule.register_operation(fwd_op)
        fwd_batch_ids[device_id, fwd_chunk_id] += 1

        bwd_batch_id_d = bwd_d_batch_ids[device_id, bwd_chunk_id]
        # Schedule backward_D
        bwd_op = Operation(bwd_batch_id_d, bwd_stage_id, "backward")
        schedule.register_operation(bwd_op)
        bwd_d_batch_ids[device_id, bwd_chunk_id] += 1

        # Create and register the overlapped operation
        overlapped_op = OverlappedOperation([fwd_op, bwd_op])
        schedule.register_overlapped_operation(overlapped_op)

        # Add the overlapped operation to the queue
        schedule.device_queues[device_id].add_operation(overlapped_op)

    # Process each device (rank in original code)
    for device_id in range(num_devices):
        # Step 1: nF0
        step_1_count = (num_devices - device_id - 1) * 2
        for _ in range(step_1_count):
            _schedule_forward_chunk(device_id, 0)  # F0

        # Step 2: nF0F1
        step_2_count = device_id + 1
        for i in range(step_2_count):
            _schedule_forward_chunk(device_id, 0)  # F0
            _schedule_forward_chunk(device_id, 1)  # F1

        # Step 3: nB1W1F1 (Use zero bubble for B1)
        step_3_count = num_devices - device_id - 1
        for _ in range(step_3_count):
            _schedule_backward_chunk(device_id, 1, enable_zb=True) # B1_D (ZB enabled)
            _schedule_backward_weight_chunk(device_id)  # W1
            _schedule_forward_chunk(device_id, 1)      # F1

        # Step 4 (Main step): nF0B1F1B0 (Overlapped F and B_D)
        num_batches = config.num_batches
        step_4_count = num_batches - num_devices * 2 + device_id + 1
        is_last_rank = (device_id == num_devices - 1) # Check if it's the last rank

        for i in range(step_4_count):
            if i == 0:
                if is_last_rank:
                    # Special handling for the first iteration on the last rank
                    # Schedule F0, B1, W1 sequentially
                    _schedule_forward_chunk(device_id, 0) # F0
                    _schedule_backward_chunk(device_id, 1, enable_zb=False) # B1_D
                else:
                    # Overlap F0 and B1
                    _schedule_forward_backward_chunk(device_id, 0, 1) # F0 + B1_D
            else:
                # Overlap F1 and B0_D
                _schedule_forward_backward_chunk(device_id, 0, 1) # F0B1
            _schedule_forward_backward_chunk(device_id, 1, 0) # 


        # Step 5: nB1F1B0
        step_5_count = num_devices - device_id - 1
        for _ in range(step_5_count):
            # Schedule B1 (B1_D + B1_W) sequentially
            _schedule_backward_chunk(device_id, 1, enable_zb=False) # B1_D + W1

            # Overlap F1 and B0
            _schedule_forward_backward_chunk(device_id, 1, 0) # F1 + B0

        # Step 6: nB1B0 (The second half of the chunks use zero bubble)
        step_6_count = device_id + 1
        enable_zb = False
        for i in range(step_6_count):
            # Determine if ZB should be enabled for B1
            if i == step_6_count // 2 and device_id % 2 == 1:
                enable_zb = True
            _schedule_backward_chunk(device_id, 1, enable_zb=enable_zb) # B1_D

            # Determine if ZB should be enabled for B0
            # ZB is enabled after the midpoint check for B0
            if i == step_6_count // 2 and device_id % 2 == 0:
                enable_zb = True # Enable ZB for the rest, including B0
            _schedule_backward_chunk(device_id, 0, enable_zb=enable_zb) # B0_D

        # Step 7: nWB0 (Use zero bubble for B0)
        step_7_count = num_devices - device_id - 1
        for _ in range(step_7_count):
            _schedule_backward_weight_chunk(device_id) # W1 (from ZB B1_D in Step 6 or Step 3)
            _schedule_backward_chunk(device_id, 0, enable_zb=True) # B0_D

        # Step 8: nW
        step_8_count = device_id + 1
        for _ in range(step_8_count):
             _schedule_backward_weight_chunk(device_id) # W0 (from ZB B0_D in Step 6 or 7) or W1 (from ZB B1_D in Step 6)

        # Final check: Ensure all waited gradients are processed
        assert not waited_weight_grad[device_id], f"Device {device_id} has remaining waited weight grads: {waited_weight_grad[device_id]}"


    return schedule
