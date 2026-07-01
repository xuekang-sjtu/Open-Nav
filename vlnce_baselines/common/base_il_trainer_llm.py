import json
import sys
import jsonlines
import os
import time
import warnings
from pathlib import Path

# Resolve project root for shared model paths (cross-platform)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
from shared.evaluation_selection import filter_ids_by_cross_floor
from shared.results import aggregate_numeric_metrics
from shared.resume_utils import append_episode_metric
from collections import defaultdict
from typing import Dict, List
from PIL import Image
import requests
from openai import OpenAI

# for navigator      
from vlnce_baselines.common.navigator.spatialNavigator import *
import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as distr
import torch.multiprocessing as mp
import gzip
import math
from copy import deepcopy

import tqdm
from gym import Space
from habitat import Config, logger
from habitat.utils.visualizations.utils import append_text_to_image
from habitat_baselines.common.base_il_trainer import BaseILTrainer
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.environments import get_env_class
from habitat_baselines.common.obs_transformers import (
    apply_obs_transforms_batch,
    apply_obs_transforms_obs_space,
    get_active_obs_transforms,
)
from habitat_extensions.measures import Position
from habitat_baselines.common.tensorboard_utils import TensorboardWriter
from habitat_baselines.utils.common import batch_obs, generate_video
from habitat_baselines.utils.common import (
    get_checkpoint_id,
    poll_checkpoint_folder,
)

from habitat_extensions.utils import observations_to_image
from vlnce_baselines.common.aux_losses import AuxLosses
from vlnce_baselines.common.env_utils import (
    construct_envs_auto_reset_false,
    construct_envs,
    is_slurm_batch_job,
)
from vlnce_baselines.common.utils import *

from habitat_extensions.measures import NDTW
from fastdtw import fastdtw

from ..utils import get_camera_orientations
from ..models.utils import (
    length2mask, dir_angle_feature, dir_angle_feature_with_ele,
)
from shared.eval_metrics import format_episode_metric
from shared.ssa import SSAController, execute_ssa_takeover
from shared.ssa.oracle import select_oracle_exit_for_episode
from shared.ssa.trajectory import save_trajectory_debug


def _ssa_front_view(images_dict):
    return images_dict.get("0")


def _ssa_view_yaw_deg(angle_value):
    angle_deg = float(np.rad2deg(angle_value))
    return ((angle_deg + 180.0) % 360.0) - 180.0


def _ssa_current_stage_from_estimation(actions, estimation):
    action_lines = [
        line.strip(" \t-0123456789.)")
        for line in str(actions or "").splitlines()
        if line.strip()
    ]
    executed_text = str(estimation or "").lower()
    for action in action_lines:
        if action and action.lower() not in executed_text:
            return action
    return action_lines[-1] if action_lines else ""

# TensorFlow import removed - not used in this codebase
# Original: with warnings.catch_warnings(): warnings.filterwarnings("ignore", category=FutureWarning); import tensorflow as tf

class BaseVLNCETrainerLLM(BaseILTrainer):
    r"""A base trainer for VLN-CE imitation learning."""
    supported_tasks: List[str] = ["VLN-v0"]

    def __init__(self, config=None):
        super().__init__(config)
        self.policy = None
        self.device = (
            torch.device("cuda", self.config.TORCH_GPU_ID)
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        self.obs_transforms = []
        self.start_epoch = 0
        self.step_id = 0

    def _initialize_policy(
        self,
        config: Config,
        load_from_ckpt: bool,
        observation_space: Space,
        action_space: Space,
    ) -> None:
        policy = baseline_registry.get_policy(self.config.MODEL.policy_name)
        self.policy = policy.from_config(
            config=config,
            observation_space=observation_space,
            action_space=action_space,
        )
        ''' initialize the waypoint predictor here '''
        from waypoint_prediction.TRM_net import BinaryDistPredictor_TRM
        self.waypoint_predictor = BinaryDistPredictor_TRM(device=self.device)
        self.waypoint_predictor.load_state_dict(
            torch.load(
                os.path.join(PROJECT_ROOT, "models", "waypoint_prediction", "checkpoints", "check_val_best_avg_wayscore"),
                map_location=torch.device('cpu'),
                weights_only=False,
            )['predictor']['state_dict']
        )
        for param in self.waypoint_predictor.parameters():
            param.requires_grad = False

  
        self.policy.to(self.device)
        self.waypoint_predictor.to(self.device)
        self.num_recurrent_layers = self.policy.net.num_recurrent_layers

        logger.info("Finished setting up waypoint_predictor.")

    def load_checkpoint(self, checkpoint_path, *args, **kwargs) -> Dict:
        return torch.load(checkpoint_path, weights_only=False, *args, **kwargs)

    @staticmethod
    def _pause_envs(
        envs_to_pause,
        envs,
        not_done_masks,
        prev_actions,
        batch,
        rgb_frames=None,
    ):
        if len(envs_to_pause) > 0:
            state_index = list(range(envs.num_envs))
            for idx in reversed(envs_to_pause):
                state_index.pop(idx)
                envs.pause_at(idx)
                
            not_done_masks = not_done_masks[state_index]
            prev_actions = prev_actions[state_index]

            for k, v in batch.items():
                batch[k] = v[state_index]

            if rgb_frames is not None:
                rgb_frames = [rgb_frames[i] for i in state_index]

        return (
            envs,
            not_done_masks,
            prev_actions,
            batch,
            rgb_frames,
        )
        
    def generate_input(self, observations):
        instruction = observations['instruction']['text']
        image_dict = {} 
        rgb_image_dict = {}
        depth_image_dict = {}
        rgb_index = 0
        depth_index = 0
        for key in observations.keys():
            image_path = "./image_show/"
            if 'rgb' in key:
                image_path += f"{key}.jpg"
                image = Image.fromarray(observations[key], mode="RGB")
                dir_name = os.path.dirname(image_path)
                if not os.path.exists(dir_name):
                    os.makedirs(dir_name)
                image.save(image_path, format="JPEG")
                rgb_image_dict[str(rgb_index)] = Image.open(image_path)
                rgb_index += 1
            if 'depth' in key:
                image_path += f"{key}.jpg"
                if observations[key].ndim == 3 and observations[key].shape[-1] == 1:
                    depth_map = observations[key].squeeze(-1)
                depth_img = (255 * (depth_map - np.min(depth_map)) / (np.max(depth_map) - np.min(depth_map))).astype(np.uint8)
                image = Image.fromarray(depth_img)
                dir_name = os.path.dirname(image_path)
                if not os.path.exists(dir_name):
                    os.makedirs(dir_name)
                image.save(image_path)
                depth_image_dict[str(depth_index)] = Image.open(image_path)
                depth_index += 1
        for index in rgb_image_dict:
            image_dict[index] = {
                'rgb': rgb_image_dict[index],
                'depth': depth_image_dict[index]
            }
            
        return instruction, image_dict
    
    def construct_image_dicts(self, batch_distance, batch_angles, image_dict):
        waypoint_distances = {}
        waypoint_radius = {}
        waypoint_images = {}
        angles = batch_angles[-1]
        for angle_idx in range(len(angles)):
            angle = angles[angle_idx]
            angle_deg = np.rad2deg(angle)
            if 0 < angle_deg <= 30:
                waypoint_images['1'] = image_dict['1']
                waypoint_distances['1'] = batch_distance[angle_idx]
                waypoint_radius['1'] = angles[angle_idx]
            elif 30 < angle_deg <= 60:
                waypoint_images['2'] = image_dict['2']
                waypoint_distances['2'] = batch_distance[angle_idx]
                waypoint_radius['2'] = angles[angle_idx]
            elif 60 < angle_deg <= 90:
                waypoint_images['3'] = image_dict['3']
                waypoint_distances['3'] = batch_distance[angle_idx]
                waypoint_radius['3'] = angles[angle_idx]
            elif 90 < angle_deg <= 120:
                waypoint_images['4'] = image_dict['4']
                waypoint_distances['4'] = batch_distance[angle_idx]
                waypoint_radius['4'] = angles[angle_idx]
            elif 120 < angle_deg <= 150:
                waypoint_images['5'] = image_dict['5']
                waypoint_distances['5'] = batch_distance[angle_idx]
                waypoint_radius['5'] = angles[angle_idx]
            elif 150 < angle_deg <= 180:
                waypoint_images['6'] = image_dict['6']
                waypoint_distances['6'] = batch_distance[angle_idx]
                waypoint_radius['6'] = angles[angle_idx]
            elif 180 < angle_deg <= 210:
                waypoint_images['7'] = image_dict['7']
                waypoint_distances['7'] = batch_distance[angle_idx]
                waypoint_radius['7'] = angles[angle_idx]
            elif 210 < angle_deg <= 240:
                waypoint_images['8'] = image_dict['8']
                waypoint_distances['8'] = batch_distance[angle_idx]
                waypoint_radius['8'] = angles[angle_idx]
            elif 240 < angle_deg <= 270:
                waypoint_images['9'] = image_dict['9']
                waypoint_distances['9'] = batch_distance[angle_idx]
                waypoint_radius['9'] = angles[angle_idx]
            elif 270 < angle_deg <= 300:
                waypoint_images['10'] = image_dict['10']
                waypoint_distances['10'] = batch_distance[angle_idx]
                waypoint_radius['10'] = angles[angle_idx]
            elif 300 < angle_deg <= 330:
                waypoint_images['11'] = image_dict['11']
                waypoint_distances['11'] = batch_distance[angle_idx]
                waypoint_radius['11'] = angles[angle_idx]
            else:
                waypoint_images['0'] = image_dict['0']  
                waypoint_distances['0'] = batch_distance[angle_idx]
                waypoint_radius['0'] = angles[angle_idx]
                
        return waypoint_images, waypoint_radius, waypoint_distances
    

    def _eval_llm(
        self,
    ) -> None:
        r"""Evaluation.

        Args:
            writer: tensorboard writer object
            checkpoint_index: index of the current checkpoint

        Returns:
            None
        """
        config = self.config.clone()


        config.defrost()
        config.TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.SHUFFLE = False
        config.TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.MAX_SCENE_REPEAT_STEPS = (
            -1
        )
        if len(config.VIDEO_OPTION) > 0:
            config.defrost()
            config.TASK_CONFIG.TASK.MEASUREMENTS.append("TOP_DOWN_MAP_VLNCE")
            config.TASK_CONFIG.TASK.MEASUREMENTS.append("COLLISIONS")
        config.freeze()

        if config.EVAL.SAVE_RESULTS:
            fname = os.path.join(
                config.RESULTS_DIR,
                f"stats_ckpt_{config.TASK_CONFIG.DATASET.SPLIT}.json",
            )
            if os.path.exists(fname):
                print(f"skipping -- evaluation exists. File path: {fname}")
                user_input = os.environ.get("OVERWRITE_RESULTS", "yes").strip().lower()
                if user_input != "yes":
                    print("Skipping evaluation.")
                    return
                else:
                    print("Overwriting previous results...")
                

        envs = construct_envs(
            config, get_env_class(config.ENV_NAME),
            auto_reset_done=False,
            episodes_allowed=self.traj
        ) 

        #envs.number_of_episodes = [1] # set the number of episodes
        dataset_length = sum(envs.number_of_episodes) 
        print('local rank:', self.local_rank, '|', 'dataset length:', dataset_length)

        obs_transforms = get_active_obs_transforms(config) 
        observation_space = apply_obs_transforms_obs_space(
            envs.observation_spaces[0], obs_transforms
        )
        self._initialize_policy(
            config,
            load_from_ckpt=False,
            observation_space=observation_space,
            action_space=envs.action_spaces[0],
        )
        self.policy.eval() 
        self.waypoint_predictor.eval()
        observations = envs.reset()
        
        instruction, images_list = self.generate_input(observations[-1])
        observations = extract_instruction_tokens(
            observations, self.config.TASK_CONFIG.TASK.INSTRUCTION_SENSOR_UUID
        ) 
        batch = batch_obs(observations, self.device) 
        batch = apply_obs_transforms_batch(batch, obs_transforms) 

        not_done_masks = torch.zeros(
            envs.num_envs, 1, dtype=torch.uint8, device=self.device
        ) 

        stats_episodes = {}
        rgb_frames = [[] for _ in range(envs.num_envs)]
        if len(config.VIDEO_OPTION) > 0:
            os.makedirs(config.VIDEO_DIR, exist_ok=True)

        if config.EVAL.EPISODE_COUNT == -1:
            episodes_to_eval = sum(envs.number_of_episodes)
        else:
            episodes_to_eval = min(
                config.EVAL.EPISODE_COUNT, sum(envs.number_of_episodes)
            )

        pbar = tqdm.tqdm(total=episodes_to_eval) if config.use_pbar else None
        log_str = (
            " [Episodes evaluated: {evaluated}/{total}]"
            " [Time elapsed (s): {time}]"
        )
        start_time = time.time()

        # set up the logger
        log_file = "./navigator_log.log"
        if os.path.exists(log_file): os.remove(log_file)
        import logging
        logging.basicConfig(
            format='%(asctime)s - %(filename)s/%(funcName)s[line:%(lineno)d] - %(levelname)s: %(message)s',
            datefmt="%Y-%m-%d %H:%M:%S",
            level=os.environ.get("LOGLEVEL", "INFO").upper(),
            stream=sys.stdout,
            filemode="a"
        )
        nav_logger = logging.getLogger("vln_logger")
        nav_logger.addHandler(logging.FileHandler(filename=log_file))
        
        dataset_name = "R2R"
        if not os.path.exists(f"cache_files/{dataset_name}"):
            os.makedirs(f"cache_files/{dataset_name}")

        actions_cache_path = f"./cache_files/{dataset_name}/actions_cache_open-nav.json"
        if os.path.exists(actions_cache_path): 
            with open(actions_cache_path, "r", encoding="utf-8") as file:
                actions_cache = json.load(file)
        else:
            actions_cache = {} 
        
        navigator = Open_Nav(self.device,config.LLM, config.API_KEY)
        ssa_controller = SSAController(
            enabled=getattr(config, "SSA_GUIDANCE", False),
            workspace_root=Path(PROJECT_ROOT),
            checkpoint_path=getattr(config, "SSA_CHECKPOINT", ""),
            detect_threshold=float(getattr(config, "SSA_DETECT_THRESHOLD", 0.5)),
            detector_model_source=getattr(config, "SSA_DETECTOR_MODEL_SOURCE", None),
            filter_behind=getattr(config, "SSA_FILTER_BEHIND", False),
            oracle_exit_enabled=getattr(config, "SSA_ORACLE_EXIT_ENABLE", False),
        )
        current_step = 0
        nav_history = []
        error_number = 0
        while envs.num_envs > 0 and len(stats_episodes) < episodes_to_eval:
            current_episodes = envs.current_episodes()
            positions = []; headings = []
            for ob_i in range(len(current_episodes)): 
                agent_state_i = envs.call_at(ob_i,
                        "get_agent_info", {})
                positions.append(agent_state_i['position'])
                headings.append(agent_state_i['heading'])
            # ==========Navigator start==========
            nav_logger.info(f"==================== The current episode id is {current_episodes[0].episode_id} ====================")
            nav_logger.info("Instruction: "+instruction)
            actions, landmarks = "", ""
            if instruction not in actions_cache.keys():
                nav_logger.info("[Cache MISS] Calling LLM to decompose instruction...")
                actions = navigator.get_actions(instruction)
                landmarks = navigator.get_landmarks(actions)
                actions_cache[instruction] = {"actions": actions, "landmarks": landmarks}
                with open(actions_cache_path, "w", encoding="utf-8") as f2:
                    json.dump(actions_cache, f2, indent=2)
                nav_logger.info("[Cache SAVED] Instruction cached to disk")
            else:
                nav_logger.info("[Cache HIT] Reusing cached instruction decomposition")
                actions = actions_cache[instruction]["actions"]
                landmarks = actions_cache[instruction]["landmarks"]
            nav_logger.info("Actions: "+actions)
            nav_logger.info("Landmarks: " + landmarks)
            
            step_length = 6 if len(actions.split("\n")) <= 6 else 8 

            stop_flag = False
            current_step += 1
            nav_logger.info(f"-------------------- Step {current_step} --------------------")
            with torch.no_grad():
                # candidate waypoints prediction
                cand_rgb, cand_depth, \
                cand_direction, cand_mask, candidate_lengths, \
                batch_angles, batch_distances = self.policy.net( 
                    mode = "waypoint",
                    waypoint_predictor = self.waypoint_predictor,
                    observations = batch,
                    in_train = False,
                )
            
            images_dict, radius_dict, distance_dict = self.construct_image_dicts(batch_distances[-1], batch_angles, images_list)
            nav_logger.info("========== Get Observation ==========")
            observation, observe_dict = navigator.observe_environment(nav_logger, current_step, images_dict)
            
            nav_logger.info("========== Review History ==========")
            history_traj = navigator.review_history(nav_logger, nav_history) if len(nav_history) > 0 else "Step 0 start position. "
            ssa_takeover_requested = False
            ssa_takeover_direction = "unknown"
            ssa_pre_align_yaw_rad = None

            if not stop_flag:
                nav_logger.info("========== Estimate Completion Progress ==========")
                estimation = navigator.estimate_completion(nav_logger, actions, landmarks, history_traj)
                
                nav_logger.info("========== Next Action Prediction ==========")
                predictions, thoughts, break_flag = navigator.move_to_next_vp(nav_logger, current_step, instruction, actions, landmarks, history_traj, estimation, observation, observe_dict)

                nav_logger.info("========== Thought ==========")
                fused_pred_thought = navigator.thought_fusion(nav_logger, predictions, thoughts)
                
                nav_logger.info("========== Test Decision ==========")
                next_vp, thought, error_number = navigator.test_decisions(nav_logger, fused_pred_thought, observation, instruction, error_number, observe_dict)
                selected_ssa_view = images_dict.get(next_vp)
                selected_ssa_yaw_deg = _ssa_view_yaw_deg(radius_dict[next_vp]) if next_vp in radius_dict else 0.0
                current_stage_text = _ssa_current_stage_from_estimation(actions, estimation)
                current_context_text = str(history_traj or "")
                if selected_ssa_view is not None:
                    ssa_proposal = ssa_controller.update_proposal(
                        instruction="",
                        previous_output=current_stage_text,
                        previous_plan="",
                        rgb=np.asarray(selected_ssa_view["rgb"]),
                        depth=np.asarray(selected_ssa_view["depth"]),
                        view_yaw_deg=selected_ssa_yaw_deg,
                        delegate_infer_fn=lambda *_: '{"delegate": false, "direction": "unknown", "reason": "unused"}',
                        delegate_image_infer_fn=navigator.llm.gpt_infer_with_images,
                        delegate_image=selected_ssa_view,
                        delegate_current_stage=current_stage_text,
                        delegate_history=current_context_text,
                        delegate_observation_hint=observe_dict.get(next_vp, ""),
                    )
                else:
                    ssa_proposal = {"available": False, "reason": "missing_selected_view"}
                ssa_controller.record_step_proposal(
                    step=current_step,
                    available=bool(ssa_proposal.get("available", False)),
                    reason=str(ssa_proposal.get("reason", "")),
                    viewpoint=next_vp,
                    view_yaw_deg=selected_ssa_yaw_deg,
                )
                nav_logger.info(
                    f"[SSA] step={current_step} episode={current_episodes[0].episode_id} "
                    f"viewpoint={next_vp} view_yaw_deg={selected_ssa_yaw_deg:.1f} "
                    f"available={bool(ssa_proposal.get('available', False))} "
                    f"reason={ssa_proposal.get('reason', '')}"
                )
                if ssa_proposal.get("available", False):
                    delegate_info = ssa_proposal.get("delegate", {}) if isinstance(ssa_proposal.get("delegate"), dict) else {}
                    delegate_reason = "vlm_fallback" if ssa_proposal.get("reason") == "delegate_vlm" else "rule_and_dino_gate"
                    ssa_takeover_requested = True
                    ssa_takeover_direction = str(ssa_proposal.get("direction", "unknown"))
                    ssa_pre_align_yaw_rad = radius_dict[next_vp] if next_vp in radius_dict else None
                    ssa_controller.record_delegate_decision(
                        step=current_step,
                        delegated=True,
                        current_stage=current_stage_text,
                        history=current_context_text,
                        observation_hint=observe_dict.get(next_vp, ""),
                        prompt_has_rgb=bool(delegate_info.get("prompt_has_rgb", False)),
                        raw_response=str(delegate_info.get("raw_response", "")),
                        reason=delegate_reason,
                        direction=ssa_takeover_direction,
                    )
                    ssa_controller.record_plan_outcome(
                        step=current_step,
                        accepted=True,
                        reason="closed_loop_ready",
                        planned_actions=0,
                    )
                    nav_logger.info(
                        f"[SSA] step={current_step} episode={current_episodes[0].episode_id} delegated=yes mode=closed_loop direction={ssa_takeover_direction}"
                    )
           
            try:
                if not stop_flag:
                    ssa_takeover_finished_episode = False
                    if ssa_takeover_requested:
                        def _ssa_get_forward_view(observation_item):
                            _, ssa_images = self.generate_input(observation_item)
                            ssa_front = ssa_images.get("0") if isinstance(ssa_images, dict) else None
                            if ssa_front is None:
                                raise RuntimeError("SSA takeover requires a forward RGB-D view")
                            return np.asarray(ssa_front["rgb"]), np.asarray(ssa_front["depth"])

                        takeover = execute_ssa_takeover(
                            envs,
                            env_index=0,
                            controller=ssa_controller,
                            initial_observation=observations[-1],
                            get_forward_view=_ssa_get_forward_view,
                            direction=ssa_takeover_direction,
                            step=current_step,
                            pre_align_yaw_rad=ssa_pre_align_yaw_rad,
                            oracle_exit=select_oracle_exit_for_episode(
                                current_episodes[0],
                                current_position=envs.call_at(0, "get_agent_info", {}).get("position"),
                                direction=ssa_takeover_direction,
                            ),
                        )
                        nav_logger.info(f"[SSA] takeover finished | success={takeover.success} reason={takeover.reason} actions={takeover.actions_executed}")
                        observations = takeover.observations
                        dones = takeover.dones
                        infos = takeover.infos
                        instruction, images_list = self.generate_input(observations[-1])
                        final_ssa_view = images_list.get("0") if isinstance(images_list, dict) else None
                        if final_ssa_view is not None:
                            _, final_observe_dict = navigator.observe_environment(
                                nav_logger,
                                current_step,
                                {"0": final_ssa_view},
                            )
                            ssa_thought = (
                                f"SSA takeover executed {takeover.actions_executed} waypoint steps; "
                                f"result={takeover.reason}."
                            )
                            nav_logger.info("========== save SSA history ==========")
                            nav_history = navigator.save_history(
                                nav_logger,
                                current_step,
                                "0",
                                ssa_thought,
                                final_observe_dict["0"],
                                nav_history,
                            )
                        else:
                            nav_logger.warning("SSA final forward view missing; history not updated")
                        observations = extract_instruction_tokens(
                            observations, self.config.TASK_CONFIG.TASK.INSTRUCTION_SENSOR_UUID
                        )
                        batch = batch_obs(observations, self.device)
                        batch = apply_obs_transforms_batch(batch, obs_transforms)
                        if not dones[0]:
                            continue
                        dones[0] = True
                        ssa_takeover_finished_episode = True
                    if not ssa_takeover_finished_episode:
                        env_actions = []
                        env_actions.append({'action':
                            {'action': 4,
                            'action_args':{
                                'angle': radius_dict[next_vp],
                                'distance': distance_dict[next_vp],
                            }}})
                        nav_logger.info(f"The final env action: {env_actions}")
                        outputs = envs.step(env_actions)
                        
                        curr_observe = observe_dict[next_vp]
                        nav_logger.info("========== save history ==========")
                        nav_history = navigator.save_history(nav_logger, current_step, next_vp, thought, curr_observe, nav_history)
                    
                        observations, _, dones, infos = [list(x) for x in zip(*outputs)]
                        instruction, images_list = self.generate_input(observations[-1])
                        error_number = 0 
                        # finish navigation
                        if current_step == step_length:
                            dones[0] = True 
                        else:
                            for j, ob in enumerate(observations):
                                envs.call_at(j, 
                                    'change_current_path',
                                    {'new_path': ob.pop('positions'),
                                    'collisions': ob.pop('collisions')}
                                )
                else:
                    dones[0] = True
                
                not_done_masks = torch.tensor(
                    [[0] if done else [1] for done in dones],
                    dtype=torch.uint8, device=self.device)
                
                for i in range(envs.num_envs):
                    
                    if not dones[i]:
                        continue
                    
                    current_step = 0
                    nav_history = []
                    ssa_trace_path = ssa_controller.save_episode_trace(config.RESULTS_DIR, current_episodes[i].episode_id)
                    ssa_summary = ssa_controller.episode_summary()
                    ssa_trace = ssa_controller.episode_trace()
                    nav_logger.info(
                        f"[SSA] episode summary | episode={current_episodes[i].episode_id} {ssa_controller.episode_summary_text()}"
                    )
                    ssa_controller.reset()
                    info = infos[i]
                    metric = {}
                    metric['steps_taken'] = info['steps_taken']
                    metric["ssa_summary"] = ssa_summary
                    metric["ssa_trace_path"] = ssa_trace_path
                    ep_id = str(envs.current_episodes()[i].episode_id)
                    gt_path = np.array(self.gt_data[ep_id]['locations']).astype(float)
                    if 'current_path' in envs.current_episodes()[i].info.keys():
                        positions_ = np.array(envs.current_episodes()[i].info['current_path']).astype(float)
                        collisions_ = np.array(envs.current_episodes()[i].info['collisions'])
                        assert collisions_.shape[0] == positions_.shape[0] - 1
                    else:
                        positions_ = np.array(dis_to_con(np.array(info['position']['position']))).astype(float)
                    distance = np.array(info['position']['distance']).astype(float)
                    metric['distance_to_goal'] = distance[-1]
                    metric['success'] = 1. if distance[-1] <= 3. else 0.
                    metric['oracle_success'] = 1. if (distance <= 3.).any() else 0.
                    metric['path_length'] = np.linalg.norm(positions_[1:] - positions_[:-1],axis=1).sum()
                    metric['collisions'] = collisions_.mean()
                    gt_length = distance[0]
                    metric['spl'] = metric['success']*gt_length/max(gt_length,metric['path_length'])

                    act_con_path = positions_
                    gt_con_path = np.array(gt_path).astype(float)
                    dtw_distance = fastdtw(act_con_path, gt_con_path, dist=NDTW.euclidean_distance)[0]
                    nDTW = np.exp(-dtw_distance / (len(gt_con_path) * config.TASK_CONFIG.TASK.SUCCESS_DISTANCE))

                    metric['ndtw'] = nDTW
                    stats_episodes[current_episodes[i].episode_id] = metric 
                    nav_logger.info(
                        format_episode_metric(
                            current_episodes[i].episode_id,
                            metric,
                            stats=stats_episodes,
                            total=episodes_to_eval,
                        )
                    )
                    append_episode_metric(
                        config.RESULTS_DIR,
                        f"episode_results_{config.TASK_CONFIG.DATASET.SPLIT}_r{self.local_rank}_w{self.world_size}.json",
                        current_episodes[i].episode_id,
                        metric,
                    )
                    ssa_trajectory = []
                    for result in ssa_trace.get("takeover_results", []) or []:
                        ssa_trajectory.extend(result.get("ssa_trajectory", []) or [])
                    save_trajectory_debug(
                        output_dir=config.RESULTS_DIR,
                        episode_id=str(current_episodes[i].episode_id),
                        payload={
                            "episode_id": str(current_episodes[i].episode_id),
                            "scene_id": current_episodes[i].scene_id,
                            "metric": metric,
                            "start_position": positions_[0].tolist() if len(positions_) else [],
                            "goal_position": gt_path[-1].tolist() if len(gt_path) else [],
                            "agent_trajectory": [
                                {"step": int(j), "source": "agent", "position": pos.tolist()}
                                for j, pos in enumerate(positions_)
                            ],
                            "ssa_trajectory": ssa_trajectory,
                            "expert_trajectory": gt_path.tolist(),
                            "ssa_trace": ssa_trace,
                        },
                    )

                    observations[i] = envs.reset_at(i)[0]
                    instruction, images_list = self.generate_input(observations[i])
                    
                    if config.use_pbar:
                        pbar.update()
                    else:
                        logger.info(
                            log_str.format(
                                evaluated=len(stats_episodes),
                                total=episodes_to_eval,
                                time=round(time.time() - start_time),
                            )
                        )
                observations = extract_instruction_tokens(
                    observations,
                    self.config.TASK_CONFIG.TASK.INSTRUCTION_SENSOR_UUID,
                )
                batch = batch_obs(observations, self.device)
                batch = apply_obs_transforms_batch(batch, obs_transforms)   
                
                envs_to_pause = []
                next_episodes = envs.current_episodes()

                for i in range(envs.num_envs):
                    if next_episodes[i].episode_id in stats_episodes:
                        envs_to_pause.append(i)

                headings = torch.tensor(headings)
                (
                    envs,
                    not_done_masks,
                    headings,  
                    batch,
                    rgb_frames,
                ) = self._pause_envs(
                    envs_to_pause,
                    envs,
                    not_done_masks,
                    headings,
                    batch,
                    rgb_frames,
                )
                headings = headings.tolist()
            except Exception as e:
                nav_logger.info(f"Error in next action prediction: {e}")
                current_step -= 1
        envs.close()
        if config.use_pbar:
            pbar.close()
        if self.world_size > 1:
            distr.barrier()
        valid_stats = [value for value in stats_episodes.values() if isinstance(value, dict)]
        num_episodes = len(valid_stats)
        if num_episodes == 0:
            logger.info("No newly evaluated episodes with metrics were produced in this run.")
            return
        aggregated_stats = aggregate_numeric_metrics(stats_episodes)
        total = torch.tensor(num_episodes).cpu()
        if self.world_size > 1:
            dist.reduce(total,dst=0)
        total = total.item()

        if self.world_size > 1:
            logger.info(
                f"rank {self.local_rank}'s {num_episodes}-episode results: {aggregated_stats}")
            for k,v in aggregated_stats.items():
                v = torch.tensor(v*num_episodes).cuda()
                cat_v = gather_list_and_concat(v,self.world_size)
                v = (sum(cat_v)/total).item()
                aggregated_stats[k] = v

        split = config.TASK_CONFIG.DATASET.SPLIT
        fname = os.path.join(
            config.RESULTS_DIR,
            f"stats_ep_ckpt_{split}_r{self.local_rank}_w{self.world_size}.json",
        )
        with open(fname, "w") as f:
            json.dump(stats_episodes, f, indent=4)

        if self.local_rank < 1:
            if config.EVAL.SAVE_RESULTS:
                fname = os.path.join(
                    config.RESULTS_DIR,
                    f"stats_ckpt_{split}.json",
                )
                with open(fname, "w") as f:
                    json.dump(aggregated_stats, f, indent=4)

            logger.info(f"Episodes evaluated: {total}")
            for k, v in aggregated_stats.items():
                logger.info(f"Average episode {k}: {v:.6f}")
        
    def collect_val_traj(self):
        trajectories = defaultdict(list)
        split = self.config.TASK_CONFIG.DATASET.SPLIT
        with gzip.open(
            self.config.TASK_CONFIG.TASK.NDTW.GT_PATH.format(
                split=split)
        ) as f:
            gt_data = json.load(f)
        self.gt_data = gt_data
        trajectories = gt_data
        self.trajectories = gt_data
        trajectories = list(trajectories.keys())[self.config.local_rank::self.config.GPU_NUMBERS]
        # Apply cross-floor filter if EPISODES_ALLOWED is explicitly set
        allowed = self.config.TASK_CONFIG.DATASET.EPISODES_ALLOWED
        if allowed is not None:
            trajectories = filter_ids_by_cross_floor(trajectories, allowed)
        return trajectories
        
    def eval(self) -> None:
        r"""Main method of trainer evaluation. 

        Returns:
            None
        """
        self.device = (
            torch.device("cuda", self.config.TORCH_GPU_ID)
            if torch.cuda.is_available()
            else torch.device("cpu")
        )

        if "tensorboard" in self.config.VIDEO_OPTION:
            assert (
                len(self.config.TENSORBOARD_DIR) > 0
            ), "Must specify a tensorboard directory for video display"
            os.makedirs(self.config.TENSORBOARD_DIR, exist_ok=True)
        if "disk" in self.config.VIDEO_OPTION:
            assert (
                len(self.config.VIDEO_DIR) > 0
            ), "Must specify a directory for storing videos on disk"

        world_size = self.config.GPU_NUMBERS
        self.world_size = world_size
        self.local_rank = self.config.local_rank

        self.config.defrost()
        self.config.TASK_CONFIG.DATASET.ROLES = ["guide"]
        self.config.TASK_CONFIG.TASK.MEASUREMENTS = ['POSITION',
                                                     'STEPS_TAKEN',
                                                     ]
        if 'HIGHTOLOW' in self.config.TASK_CONFIG.TASK.POSSIBLE_ACTIONS:
            idx = self.config.TASK_CONFIG.TASK.POSSIBLE_ACTIONS.index('HIGHTOLOW')
            self.config.TASK_CONFIG.TASK.POSSIBLE_ACTIONS[idx] = 'HIGHTOLOWEVAL'
        self.config.TASK_CONFIG.DATASET.LANGUAGES = self.config.EVAL.LANGUAGES
        self.config.TASK_CONFIG.DATASET.SPLIT = self.config.EVAL.SPLIT
        self.config.TASK_CONFIG.TASK.NDTW.SPLIT = self.config.EVAL.SPLIT
        self.config.TASK_CONFIG.TASK.SDTW.SPLIT = self.config.EVAL.SPLIT
        self.config.use_pbar = not is_slurm_batch_job()
        if 'rxr' in self.config.BASE_TASK_CONFIG_PATH:
            self.config.EVAL.trajectories_file = \
                self.config.EVAL.trajectories_file[:-8] + '_w' + \
                str(self.world_size) + '_r' + str(self.local_rank) + '.json.gz'
        
        # if choosing image
        resize_config = self.config.RL.POLICY.OBS_TRANSFORMS.RESIZER_PER_SENSOR.SIZES
        config = self.config.TASK_CONFIG
        camera_orientations = get_camera_orientations(12)

        # sensor_uuids = []
        for sensor_type in ["RGB", "DEPTH"]:
            resizer_size = dict(resize_config)[sensor_type.lower()]
            sensor = getattr(config.SIMULATOR, f"{sensor_type}_SENSOR")
            for action, orient in camera_orientations.items():
                camera_template = f"{sensor_type}_{action}"
                camera_config = deepcopy(sensor)
                camera_config.ORIENTATION = camera_orientations[action]
                camera_config.UUID = camera_template.lower()
                # sensor_uuids.append(camera_config.UUID)
                setattr(config.SIMULATOR, camera_template, camera_config)
                config.SIMULATOR.AGENT_0.SENSORS.append(camera_template)
                resize_config.append((camera_template.lower(), resizer_size))
        self.config.RL.POLICY.OBS_TRANSFORMS.RESIZER_PER_SENSOR.SIZES = resize_config
        self.config.TASK_CONFIG = config
        self.config.SENSORS = config.SIMULATOR.AGENT_0.SENSORS
        
        self.config.freeze()
        torch.cuda.set_device(self.device)
        if world_size > 1:
            distr.init_process_group(backend='nccl', init_method='env://')
            self.device = self.config.TORCH_GPU_IDS[self.local_rank]
            torch.cuda.set_device(self.device)
            self.config.defrost()
            self.config.TORCH_GPU_ID = self.config.TORCH_GPU_IDS[self.local_rank]
            self.config.freeze()
            
        self.traj = self.collect_val_traj()
        self._eval_llm()
