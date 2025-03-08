from collections import defaultdict
from src.execution_model import Schedule, ScheduleConfig


def generate_1f1b_schedule(config: ScheduleConfig):
    schedule = Schedule(config)

    for i in range(config.num_devices):
        fwd_batch_id = 0
        bwd_batch_id = 0
        cooldown_batches = warmup_batches = config.num_devices - i - 1
        steady_batches = config.num_batches - warmup_batches

        for _ in range(warmup_batches):
            for j in range(len(schedule.dev_queues[i].stages)):
                schedule.dev_queues[i].add_operation(
                    schedule.get_op(fwd_batch_id, schedule.dev_queues[i].stages[j], "forward")
                )
            fwd_batch_id += 1

        for _ in range(steady_batches):
            for j in range(len(schedule.dev_queues[i].stages)):
                schedule.dev_queues[i].add_operation(
                    schedule.get_op(fwd_batch_id, schedule.dev_queues[i].stages[j], "forward")
                )
            fwd_batch_id += 1
            for j in range(len(schedule.dev_queues[i].stages)-1, -1, -1):
                schedule.dev_queues[i].add_operation(
                    schedule.get_op(bwd_batch_id, schedule.dev_queues[i].stages[j], "backward")
                )
            bwd_batch_id += 1

        for _ in range(cooldown_batches):
            for j in range(len(schedule.dev_queues[i].stages)-1, -1, -1):
                schedule.dev_queues[i].add_operation(
                    schedule.get_op(bwd_batch_id, schedule.dev_queues[i].stages[j], "backward")
                )
            bwd_batch_id += 1

    return schedule


# Some codes are copied from Megatron-LM
def generate_1f1b_interleave_schedule(config: ScheduleConfig):
    schedule = Schedule(config)

    def get_pp_rank_microbatches(
        num_microbatches, 
        num_devices,
        device_id,
        num_stages_per_device, 
        microbatch_group_size_per_vp_stage, 
    ):
        """Get the number of total, warmup, and remaining microbatches in PP scheduling."""
        total_num_microbatches = num_microbatches * num_stages_per_device
        are_all_microbatches_in_warmup = False

        if num_devices > 1:
            if num_stages_per_device is None:
                # forward_backward_pipelining_without_interleaving
                num_warmup_microbatches = num_devices - device_id - 1
            else:
                # forward_backward_pipelining_with_interleaving
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
            are_all_microbatches_in_warmup = True
        num_microbatches_remaining = total_num_microbatches - num_warmup_microbatches

        return (
            total_num_microbatches,
            are_all_microbatches_in_warmup,
            num_warmup_microbatches,
            num_microbatches_remaining,
        )


    def get_schedule_table(num_microbatches, num_model_chunks, microbatch_group_size_per_vp_stage):
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
    

    def convert_schedule_table_to_order(num_warmup_microbatches, num_model_chunks, schedule_table):
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
    
    for device_id in range(config.num_devices):
        microbatch_group_size_per_vp_stage = config.num_devices
        total_num_microbatches, are_all_microbatches_in_warmup, num_warmup_microbatches, num_microbatches_remaining = get_pp_rank_microbatches(
            config.num_batches,
            config.num_devices,
            device_id,
            config.num_stages_per_device,
            microbatch_group_size_per_vp_stage,
        )

        schedule_table = get_schedule_table(
            config.num_batches,
            config.num_stages_per_device,
            microbatch_group_size_per_vp_stage,
        )

        order = convert_schedule_table_to_order(
            num_warmup_microbatches,
            num_model_chunks=config.num_stages_per_device, 
            schedule_table=schedule_table,
        )

        cur_stage_microbatch_id = {}
        for i in range(1, config.num_stages_per_device+1):
            cur_stage_microbatch_id[i] = 0
            cur_stage_microbatch_id[-i] = 0
        for order_item in order:
            stage_id = schedule.dev_queues[device_id].stages[abs(order_item)-1]

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
            schedule.dev_queues[device_id].add_operation(
                schedule.get_op(micro_batch_id, stage_id, op_type)
            )
    return schedule
