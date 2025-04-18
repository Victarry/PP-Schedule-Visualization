# PP schedule config
from src.execution_model import ScheduleConfig
from src.strategies import generate_dualpipe_v_schedule


p = 4 # PP size
v = 2 # number of virtual stages
m = 10 # total microbatches

# stage time config
F = 2.0 # forward time in one PP rank for all stages
W = 2.0 # backward_W time in one PP rank for all stages
D = 2.0 # backward_D time in one PP rank for all stages
B = W + D # backward time in one PP rank for all stages
FwB = 6 # overlapped forward backward time in one PP rank for all stages

op_times = {
  "forward": F,
  "backward": B,
  "backward_D": D,
  "backward_W": W,
  "overlapped_forward_backward": FwB
}

def dualpipe_v_execution_time_by_formula():
    # Formula from the image
    item_1 = ((p - 1) / 2) * F
    item_2 = (p + 0.5) * F + (p / 2 + 1) * B
    item_3 = (m - (p / 2 + 1)) * FwB
    print(f"item_1: {item_1}, item_2: {item_2}, item_3: {item_3}")
    total_time = item_1 + item_2 + item_3
    return total_time

def dualpipe_v_execution_time_by_formula_detailed():
    # Correct formula
    local_F = F / 2
    local_B = B / 2
    local_D = D / 2
    local_W = W / 2
    local_FwB = FwB / 2

    forward_bubble = (p - 1) * local_F # forward bubble
    forward_time = 2 * p * local_F
    overlapped_time = (2 * (m-p)-1) * local_FwB + (p-1) * local_FwB
    backward_time = (2*p-1) * local_D + local_W
    other_time = 2 * local_B + local_F

    active_time = (2 * (m-p)-1) * local_FwB + (2*p+1) * (local_F + local_B)
    total_time = forward_bubble + forward_time + overlapped_time + backward_time + other_time
    bubble_time = total_time - active_time

    assert bubble_time == (p-1)*(local_FwB + local_B - 3*local_W)

    return total_time

def dualpipe_v_execution_time_by_emulate():
    op_times_per_stage = {
        "forward": F / 2,
        "backward": B / 2,
        "backward_D": D / 2,
        "backward_W": W / 2,
        "overlapped_forward_backward": FwB / 2
    }
    print(f"op_times_per_stage: {op_times_per_stage}")
    dualpipe_schedule_config = ScheduleConfig(
        num_devices=p,
        num_stages=p*2,
        num_batches=m,
        p2p_latency=0.0,
        op_times=op_times_per_stage,
        split_backward=True,
        placement_strategy="dualpipe_v",
    )

    dual_pipe_schedule = generate_dualpipe_v_schedule(dualpipe_schedule_config)

    dual_pipe_schedule.execute()

    return dual_pipe_schedule.get_total_execution_time()

print(f"DualPipe-V by emulate: {dualpipe_v_execution_time_by_emulate()}")
print(f"DualPipe-V by formula detailed: {dualpipe_v_execution_time_by_formula_detailed()}")
