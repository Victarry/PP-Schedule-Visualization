from src.execution_model import ScheduleConfig, ScheduleExecutor
from src.strategies import generate_1f1b_interleave_schedule, generate_1f1b_schedule
from src.visualizer import visualize_pipeline_parallelism_dash
import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """Run pipeline parallelism simulation with the specified configuration."""
    print(f"Running with configuration: {cfg}")

    if cfg.strategy == "1f1b":
        run_1f1b(cfg)
    elif cfg.strategy == "interleave":
        run_interleave(cfg)
    else:
        raise ValueError(f"Unknown strategy: {cfg.strategy}")


def run_1f1b(cfg: DictConfig) -> None:
    """Run 1F1B pipeline parallelism simulation."""
    # Convert OmegaConf to dict for op_times if it exists
    op_times = OmegaConf.to_container(cfg.op_times) if hasattr(cfg, 'op_times') else None

    schedule_config = ScheduleConfig(
        num_devices=cfg.num_devices,
        num_stages=cfg.num_stages,
        num_batches=cfg.num_batches,
        p2p_latency=cfg.p2p_latency,
        op_times=op_times,
        placement_strategy="1f1b"
    )
    schedule = generate_1f1b_schedule(schedule_config)
    executor = ScheduleExecutor(schedule)
    executor.execute()

    visualize_pipeline_parallelism_dash(schedule, port=cfg.visualization_port)


def run_interleave(cfg: DictConfig) -> None:
    """Run interleaved pipeline parallelism simulation."""
    # Convert OmegaConf to dict for op_times if it exists
    op_times = OmegaConf.to_container(cfg.op_times) if hasattr(cfg, 'op_times') else None
    
    schedule_config = ScheduleConfig(
        num_devices=cfg.num_devices,
        num_stages=cfg.num_stages,
        num_batches=cfg.num_batches,
        p2p_latency=cfg.p2p_latency,
        placement_strategy="interleave",
        op_times=op_times
    )
    schedule = generate_1f1b_interleave_schedule(schedule_config)
    executor = ScheduleExecutor(schedule)
    executor.execute()

    visualize_pipeline_parallelism_dash(schedule, port=cfg.visualization_port)


if __name__ == "__main__":
    main()