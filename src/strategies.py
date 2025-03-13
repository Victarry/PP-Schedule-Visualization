from collections import defaultdict
from src.execution_model import OverlappedOperation, Schedule, ScheduleConfig


def generate_1f1b_schedule(config: ScheduleConfig):
    schedule = Schedule(config)

    assert config.num_devices == config.num_stages, "num_devices must be equal to num_stages for 1F1B"

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
    assert config.num_devices == config.num_stages, "num_devices must be equal to num_stages for ZB-1P"

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

    assert config.num_devices == config.num_stages, "num_devices must be equal to num_stages for 1F1B"

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
        num_warmup_microbatches += (num_stages_per_device - 1) * microbatch_group_size_per_vp_stage
    else:
        # forward_backward_no_pipelining
        num_warmup_microbatches = 1

    if num_warmup_microbatches >= total_num_microbatches:
        num_warmup_microbatches = total_num_microbatches

    return num_warmup_microbatches


def _get_schedule_table(num_microbatches, num_model_chunks, microbatch_group_size_per_vp_stage):
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
        if min_microbatch_id_in_group + microbatch_group_size_per_vp_stage >= num_microbatches:
            # Construct schedule for the last microbatch group
            schedule_table.extend(
                [
                    (microbatch_id, model_chunk_id)
                    for model_chunk_id in range(num_model_chunks)
                    for microbatch_id in range(min_microbatch_id_in_group, num_microbatches)
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


def _convert_schedule_table_to_order(num_warmup_microbatches, num_model_chunks, schedule_table):
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

        order = _convert_schedule_table_to_order(
            num_warmup_microbatches,
            num_model_chunks=config.num_stages_per_device, 
            schedule_table=schedule_table,
        )

        cur_stage_microbatch_id = {}
        for i in range(1, config.num_stages_per_device+1):
            cur_stage_microbatch_id[i] = 0
            cur_stage_microbatch_id[-i] = 0
        for order_item in order:
            stage_id = schedule.device_queues[device_id].stages[abs(order_item)-1]

            if order_item > 0:
                op_type = "forward"
                micro_batch_id = cur_stage_microbatch_id[order_item]
                cur_stage_microbatch_id[order_item] = cur_stage_microbatch_id[order_item] + 1
            elif order_item < 0:
                op_type = "backward"
                micro_batch_id = cur_stage_microbatch_id[order_item]
                cur_stage_microbatch_id[order_item] = cur_stage_microbatch_id[order_item] + 1
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
        for i in range(1, config.num_stages_per_device+1):
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
                cur_stage_microbatch_id[order_item] = cur_stage_microbatch_id[order_item] + 1

                stage_id = schedule.device_queues[device_id].stages[abs(order_item)-1]
                schedule.device_queues[device_id].add_operation(
                    schedule.get_op(micro_batch_id, stage_id, op_type)
                )
                i += 1
            elif i >= num_warmup_microbatches and i < num_warmup_microbatches + num_overlapped_batches - 1:
                order_item_a = order[i]
                order_item_b = order[i+1]

                op_type_a = "forward" if order_item_a > 0 else "backward"
                micro_batch_id_a = cur_stage_microbatch_id[order_item_a]
                cur_stage_microbatch_id[order_item_a] = cur_stage_microbatch_id[order_item_a] + 1

                op_type_b = "forward" if order_item_b > 0 else "backward"
                micro_batch_id_b = cur_stage_microbatch_id[order_item_b]
                cur_stage_microbatch_id[order_item_b] = cur_stage_microbatch_id[order_item_b] + 1

                stage_id_a = schedule.device_queues[device_id].stages[abs(order_item_a)-1]
                stage_id_b = schedule.device_queues[device_id].stages[abs(order_item_b)-1]

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
                cur_stage_microbatch_id[order_item] = cur_stage_microbatch_id[order_item] + 1

                stage_id = schedule.device_queues[device_id].stages[abs(order_item)-1]
                schedule.device_queues[device_id].add_operation(
                    schedule.get_op(micro_batch_id, stage_id, op_type)
                )
                i += 1
            

    return schedule
