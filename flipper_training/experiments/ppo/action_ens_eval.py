import torch
import contextlib
import pickle
from tqdm import tqdm
import os
from flipper_training import ROOT
from omegaconf import OmegaConf
from flipper_training.experiments.ppo.eval import log_from_eval_rollout, PPOExperimentConfig, prepare_env
from torchrl.envs import StepCounter, VecNorm, TransformedEnv, set_exploration_type
from flipper_training.utils.logutils import get_terminal_logger
from copy import deepcopy

policies = [
    "final_coarse_gaussian_terrain_thesis_98_2025-05-12_12-07-12",
    "final_mixed_objective_training_666_2025-05-16_19-08-25",
    "mixed_training_smoothness_2025-05-20_06-47-01",
]

p_root = ROOT / "runs/ppo"
LOGGER = get_terminal_logger("action_space_ensemble_eval")

CONFIGS_DIR = ROOT / "cross_eval_configs"
RESULTS_DIR = ROOT / "cross_eval_results"
with open(CONFIGS_DIR / "cross_eval_seeds.txt", "r") as f:
    SEEDS = [int(line.strip()) for line in f.readlines()]
NUM_ENVS_PER_EVAL = 16  # robots in parallel
MAX_EVAL_STEPS = 1000
OBS_NOISE = 1e-2


class ActionSpacePolicyEnsemble:
    def __init__(
        self,
        actor_value_wrapper,
        vecnorm_obj: VecNorm,
        weights: list,
        vecnorm_weights: list,
        device,
    ):  # MPPI hyperparameters
        self.policies_callables = []
        self.num_policy_options = len(weights)
        for i in range(self.num_policy_options):
            aw = deepcopy(actor_value_wrapper)
            aw.load_state_dict(weights[i])
            policy = aw.get_policy_operator()
            policy = policy.to(device)
            policy.eval()
            vecnorm = deepcopy(vecnorm_obj)
            vecnorm.load_state_dict(vecnorm_weights[i])
            vecnorm = vecnorm.to(device)
            vecnorm.eval()
            self.policies_callables.append(lambda td: policy(vecnorm(td)))

    def __call__(self, tensordict):
        actions = []
        for p in self.policies_callables:
            actions.append(p(tensordict).pop("action"))
        actions = torch.stack(actions, dim=-1).mean(dim=-1)
        tensordict.set("action", actions)
        return tensordict


def get_action_space_ensemble(
    config: PPOExperimentConfig,
):
    env, device, rng = prepare_env(config, mode="eval")
    env = TransformedEnv(env, StepCounter())
    p_sds = [torch.load(p_root / f"{policy}/weights/policy_final.pth", map_location=device) for policy in policies]
    vecnorm_sds = [torch.load(p_root / f"{policy}/weights/vecnorm_final.pth", map_location=device) for policy in policies]
    policy_config = config.policy_config(**config.policy_opts)
    actor_value_wrapper, optim_groups, policy_transforms = policy_config.create(
        env=env,
        device=device,
    )
    vecnorm = VecNorm(in_keys=[o.name for o in env.observations if o.supports_vecnorm], **config.vecnorm_opts)
    vecnorm = vecnorm.to(device)
    planner = ActionSpacePolicyEnsemble(
        actor_value_wrapper=actor_value_wrapper,
        vecnorm_obj=vecnorm,
        weights=p_sds,
        vecnorm_weights=vecnorm_sds,
        device=device,
    )
    return planner


def get_eval_rollout(
    config: PPOExperimentConfig,
):
    env, device, rng = prepare_env(config, mode="eval")
    env = TransformedEnv(env, StepCounter())
    p_sds = [torch.load(p_root / f"{policy}/weights/policy_final.pth", map_location=device) for policy in policies]
    vecnorm_sds = [torch.load(p_root / f"{policy}/weights/vecnorm_final.pth", map_location=device) for policy in policies]
    policy_config = config.policy_config(**config.policy_opts)
    actor_value_wrapper, optim_groups, policy_transforms = policy_config.create(
        env=env,
        device=device,
    )
    vecnorm = VecNorm(in_keys=[o.name for o in env.observations if o.supports_vecnorm], **config.vecnorm_opts)
    vecnorm = vecnorm.to(device)
    env._set_truncate_mode(False)
    set_exploration_type("DETERMINISTIC")
    planner = ActionSpacePolicyEnsemble(
        actor_value_wrapper=actor_value_wrapper,
        vecnorm_obj=vecnorm,
        weights=p_sds,
        vecnorm_weights=vecnorm_sds,
        device=device,
    )
    rollout = env.rollout(config.max_eval_steps, planner, break_when_all_done=True, break_when_any_done=False)
    return env, rollout


configs = ["file_natural_cracks.yaml", "file_natural.yaml", "file_ridge.yaml", "file_sine.yaml"]


def action_space_ensembling_eval():
    FULL_RESULT_DIR = RESULTS_DIR / "ensembled_in_action_space"
    FULL_RESULT_DIR.mkdir(parents=True, exist_ok=True)
    for cfg in configs:
        test_config = OmegaConf.load(CONFIGS_DIR / cfg)
        train_config = PPOExperimentConfig(**test_config)
        train_config.num_robots = NUM_ENVS_PER_EVAL
        train_config.max_eval_steps = MAX_EVAL_STEPS
        train_config.objective_opts["cache_size"] = 10
        for obs in train_config.observations:
            if "opts" in obs:
                obs["opts"]["apply_noise"] = True
                obs["opts"]["noise_scale"] = OBS_NOISE
                LOGGER.info(f"Adding noise to {obs['cls']} with scale {OBS_NOISE}")
        devnull_handle = open(os.devnull, "w")
        results = []
        for seed in tqdm(SEEDS, desc=f"Evaluating {cfg}"):
            train_config.seed = seed
            with contextlib.redirect_stdout(devnull_handle):
                with contextlib.redirect_stderr(devnull_handle):
                    env, rollout = get_eval_rollout(train_config)
            log = log_from_eval_rollout(rollout)
            del env
            del rollout
            results.append(log)
        savename = cfg.split(".")[0]
        with open(FULL_RESULT_DIR / f"{savename}.pkl", "wb") as f:
            pickle.dump(results, f)


if __name__ == "__main__":
    action_space_ensembling_eval()
