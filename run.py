#!/usr/bin/env python3
import os
import sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
SHARED_ROOT = os.path.join(PROJECT_ROOT, "shared")
if SHARED_ROOT not in sys.path:
    sys.path.insert(0, SHARED_ROOT)
import torch
import random
import argparse
import numpy as np
from habitat import logger
import habitat_extensions  # noqa: F401
import vlnce_baselines     # noqa: F401
from vlnce_baselines.config.default import get_config
from habitat_baselines.common.baseline_registry import baseline_registry
from shared.benchmark_runner import (
    apply_cross_floor_filter_to_config,
    apply_requested_episode_filter_to_config,
    apply_resume_filter_to_config,
)
from shared.episode_filter import parse_episode_ids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp_name",
        type=str,
        default="test",
        required=True,
        help="experiment id that matches to exp-id in Notion log",
    )
    parser.add_argument(
        "--exp-config",
        type=str,
        required=True,
        help="path to config yaml containing info about experiment",
    )
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="Modify config options from command line",
    )
    parser.add_argument('--local_rank', type=int, default=0, help="local gpu id")
    parser.add_argument(
        "--llm",
        type=str,
        required=True,
        help="The LLM model to be used (e.g., gpt-4o-2024-08-06, Qwen/Qwen2.5-1.5B)",
    )
    parser.add_argument(
        "--api_key",
        type=str,
        required=True,
        help="API key for accessing the LLM service",
    )
    parser.add_argument(
        "--cross-floor-filter",
        type=str,
        default=None,
        choices=["r2r-100", "r2r-100-0.5", "r2r-all", "rxr-100", "rxr-all"],
        help="Only run cross-floor episodes",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from already completed episodes in the current experiment directory.",
    )
    parser.add_argument(
        "--episode-id",
        type=str,
        default=None,
        help="Comma-separated episode ids to evaluate, e.g. 1413,1370,1371.",
    )
    parser.add_argument("--ssa-guidance", action="store_true", help="Enable SSA stair takeover.")
    parser.add_argument("--ssa-checkpoint", type=str, default="")
    parser.add_argument("--ssa-detect-threshold", type=float, default=0.30)
    parser.add_argument("--ssa-detector-model-source", type=str, default="")
    parser.add_argument("--oracle-exit-enable", action="store_true", help="Use expert-path oracle exit as SSA diagnostic fallback.")
    parser.add_argument("--filter-behind", action="store_true", help="Reject SSA proposals where the predicted target is behind the agent.")
    args = parser.parse_args()

    # Filter out our custom arguments from opts to avoid config errors
    filtered_opts = []
    if args.opts:
        i = 0
        while i < len(args.opts):
            if args.opts[i] in ('--cross-floor-filter', '--episode-id'):
                i += 2
            else:
                filtered_opts.append(args.opts[i])
                i += 1
    args.opts = filtered_opts
    run_exp(**vars(args))
    
def run_exp(exp_name: str, exp_config: str,
            opts=None, local_rank=None,
            llm: str = None, api_key: str = None,
            episodes_to_load: int = 0, cross_floor_filter: str = None,
            resume: bool = False, ssa_guidance: bool = False,
            ssa_checkpoint: str = "", ssa_detect_threshold: float = 0.30,
            ssa_detector_model_source: str = "", filter_behind: bool = False,
            oracle_exit_enable: bool = False, episode_id: str = None) -> None:
    r"""Runs experiment given mode and config
    """
    config = get_config(exp_config, opts)
    config.defrost()

    config.CHECKPOINT_FOLDER += exp_name
    if os.path.isdir(config.EVAL_CKPT_PATH_DIR):
        config.EVAL_CKPT_PATH_DIR += exp_name
    config.RESULTS_DIR += exp_name
    config.LOG_FILE = exp_name + '_' + config.LOG_FILE

    config.TASK_CONFIG.SEED = 0
    config.local_rank = local_rank

    if llm is not None:
        config.LLM = llm
    if api_key is not None:
        config.API_KEY = api_key
    if ssa_guidance and not str(ssa_checkpoint or "").strip():
        raise ValueError("--ssa-guidance requires an explicit --ssa-checkpoint")
    config.SSA_GUIDANCE = bool(ssa_guidance)
    config.SSA_CHECKPOINT = str(ssa_checkpoint)
    config.SSA_DETECT_THRESHOLD = float(ssa_detect_threshold)
    config.SSA_DETECTOR_MODEL_SOURCE = str(ssa_detector_model_source)
    config.SSA_FILTER_BEHIND = bool(filter_behind)
    config.SSA_ORACLE_EXIT_ENABLE = bool(oracle_exit_enable)

    if cross_floor_filter is not None:
        allowed = apply_cross_floor_filter_to_config(config, cross_floor_filter)
        if cross_floor_filter == "r2r-all":
            print(f"Using R2R v1-3 dataset for cross-floor filter [{cross_floor_filter}]")
        print(f"Cross-floor filter [{cross_floor_filter}]: {len(allowed)} episodes")

    requested_episode_ids = parse_episode_ids(episode_id)
    if requested_episode_ids:
        before, after = apply_requested_episode_filter_to_config(
            config,
            requested_episode_ids,
            source="habitat",
        )
        print(
            f"Episode-id filter [{','.join(requested_episode_ids)}]: "
            f"{before} -> {after} episodes"
        )

    if resume:
        result = apply_resume_filter_to_config(config, config.RESULTS_DIR, source="habitat")
        if result is not None:
            before, after = result
            print(
                f"Resume filter: {before} -> {after} episodes "
                f"(skipped {before - after} completed from {config.RESULTS_DIR})"
            )
            if after == 0:
                print("Resume filter found no remaining episodes. Nothing to run.")
                config.freeze()
                return

    config.freeze()
    
    # Check if the 'logs/running_log' directory exists; if not, create it
    log_dir = 'logs/running_log'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Add the file handler for logging
    logger.add_filehandler(os.path.join(log_dir, config.LOG_FILE))

    random.seed(config.TASK_CONFIG.SEED)
    np.random.seed(config.TASK_CONFIG.SEED)
    torch.manual_seed(config.TASK_CONFIG.SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False
    if torch.cuda.is_available():
        torch.set_num_threads(1)

    trainer_init = baseline_registry.get_trainer(config.TRAINER_NAME)
    assert trainer_init is not None, f"{config.TRAINER_NAME} is not supported"
    trainer = trainer_init(config)
    trainer.eval()
    
if __name__ == "__main__":
    __spec__ = None 
    main()
