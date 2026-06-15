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
from resume_utils import collect_completed_episode_ids, filter_remaining_episode_ids
from episode_filter import filter_episode_ids, parse_episode_ids

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
        choices=["r2r-100", "r2r-all", "rxr-100", "rxr-all"],
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
    parser.add_argument("--ssa-checkpoint", type=str, default="SemanticSpatialAlignmentModule/outputs/20260604_121042/best_model.pt")
    parser.add_argument("--ssa-detect-threshold", type=float, default=0.30)
    parser.add_argument("--ssa-detector-model-source", type=str, default="")
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
            ssa_detector_model_source: str = "", episode_id: str = None) -> None:
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
    config.SSA_GUIDANCE = bool(ssa_guidance)
    config.SSA_CHECKPOINT = str(ssa_checkpoint)
    config.SSA_DETECT_THRESHOLD = float(ssa_detect_threshold)
    config.SSA_DETECTOR_MODEL_SOURCE = str(ssa_detector_model_source)

    if cross_floor_filter is not None:
        from cross_floor_filter import get_cross_floor_episode_ids
        allowed = get_cross_floor_episode_ids(cross_floor_filter)
        config.TASK_CONFIG.DATASET.EPISODES_ALLOWED = allowed
        print(f"Cross-floor filter [{cross_floor_filter}]: {len(allowed)} episodes")

    requested_episode_ids = parse_episode_ids(episode_id)
    if requested_episode_ids:
        current_allowed = config.TASK_CONFIG.DATASET.EPISODES_ALLOWED
        if current_allowed is None:
            import habitat
            dataset = habitat.make_dataset(
                config.TASK_CONFIG.DATASET.TYPE, config=config.TASK_CONFIG.DATASET
            )
            current_allowed = [ep.episode_id for ep in dataset.episodes]
        before = len(current_allowed)
        allowed = filter_episode_ids(current_allowed, requested_episode_ids)
        config.TASK_CONFIG.DATASET.EPISODES_ALLOWED = allowed
        print(
            f"Episode-id filter [{','.join(requested_episode_ids)}]: "
            f"{before} -> {len(allowed)} episodes"
        )

    if resume:
        completed_ids = collect_completed_episode_ids(config.RESULTS_DIR)
        if completed_ids:
            current_allowed = config.TASK_CONFIG.DATASET.EPISODES_ALLOWED
            if current_allowed is None:
                import habitat
                dataset = habitat.make_dataset(
                    config.TASK_CONFIG.DATASET.TYPE, config=config.TASK_CONFIG.DATASET
                )
                current_allowed = [ep.episode_id for ep in dataset.episodes]
            remaining = filter_remaining_episode_ids(current_allowed, completed_ids)
            before = len(current_allowed)
            config.TASK_CONFIG.DATASET.EPISODES_ALLOWED = remaining
            print(
                f"Resume filter: {before} -> {len(remaining)} episodes "
                f"(skipped {before - len(remaining)} completed from {config.RESULTS_DIR})"
            )
            if not remaining:
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
