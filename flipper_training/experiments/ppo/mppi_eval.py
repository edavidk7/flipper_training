from pathlib import Path
import matplotlib
import torch
import contextlib
import pickle
from tqdm import tqdm
import os
import torch.nn.functional as F
import torch.distributions as D
from flipper_training import ROOT
import matplotlib.pyplot as plt
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


p_root = Path("runs/ppo/")
LOGGER = get_terminal_logger("mppi_eval")

CONFIGS_DIR = ROOT / "cross_eval_configs"
RESULTS_DIR = ROOT / "cross_eval_results"
with open(CONFIGS_DIR / "cross_eval_seeds.txt", "r") as f:
    SEEDS = [int(line.strip()) for line in f.readlines()]
NUM_ENVS_PER_EVAL = 16  # robots in parallel
MAX_EVAL_STEPS = 1000
OBS_NOISE = 1e-2

steps = 30
num_samples = 100
lambda_ = 1.0
noise_sigma = 0.1
temperature = 1.0


class IteratedWeightedPolicy:
    def __init__(self, weights, policy_callables):
        self.weights = weights
        self.policy_callables = policy_callables
        self.i = 0

    def __call__(self, tensordict):
        weight_i = self.weights[self.i]  # (batch_size, num_policy_options)
        # actions: (num_policy_options, batch_size, action_dim)
        actions = []
        for p in self.policy_callables:
            actions.append(p(tensordict).pop("action"))
        actions = torch.stack(actions, dim=-1)  # shape (batch_size, action_dim, num_policy_options)
        # Compute the weighted sum of actions
        action = (actions * weight_i.unsqueeze(1)).sum(dim=-1)
        self.i += 1
        tensordict.set("action", action)
        return tensordict


class MultiPolicyMPPIPlanner:
    def __init__(
        self,
        actor_value_wrapper,
        vecnorm_obj: VecNorm,
        weights: list,
        vecnorm_weights: list,
        env,  # Not directly used in these specific methods
        device,
        rng,  # rng is expected to be a torch.Generator or compatible
        rollout_steps=20,
        num_samples=10,
        lambda_=1.0,
        noise_sigma=0.1,
        temperature=1.0,
    ):  # MPPI hyperparameters
        self.env = env
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
        self.device = device
        self.rng = rng  # Expected to be a torch.Generator for torch.randn
        self.rollout_steps = rollout_steps
        self.num_samples = num_samples
        # MPPI specific parameters
        self.lambda_ = lambda_  # Cost scaling factor in importance weights
        self.noise_sigma = noise_sigma  # Standard deviation of Gaussian noise for logits
        self.temperature = temperature  # Temperature for softmax conversion of logits to policy weights
        self.mean_logits = torch.zeros(self.rollout_steps, self.env.n_robots, self.num_policy_options, device=self.device)
        self.last_logit_perturbations = None

    def __call__(self, tensordict):
        weights = self._get_rollout_weights()
        costs = torch.empty(self.num_samples, self.env.n_robots, device=self.device)
        step = tensordict.pop("_step", None)
        for s in range(self.num_samples):
            weight = weights[s]  # (rollout_steps, batch_size, num_policy_options)
            weighted_policy = IteratedWeightedPolicy(weight, self.policies_callables)
            with set_exploration_type("DETERMINISTIC"), torch.inference_mode():
                rollout = self.env.rollout(
                    auto_reset=False,
                    policy=weighted_policy,
                    tensordict=tensordict.clone(),
                    max_steps=self.rollout_steps,
                    break_when_any_done=False,  # never break early
                    break_when_all_done=True,
                )
                rew = rollout["next", "reward"].sum(dim=[0, 1])  # (batch_size,)
                costs[s] = -rew
        self.update_mean_logits(costs)  # shape (num_samples, batch_size)
        action = self._get_optimal_action(tensordict)
        tensordict.set("action", action)
        if step is not None:
            tensordict.set("_step", step)
        return tensordict

    def _get_optimal_action(self, tensordict):
        """
        Returns the optimal action for the first step of the current
        optimized mean trajectory for each batch element.

        Args:
            tensordict: The current observation/state. Not directly used for sampling
                        logic here but is standard for planners. It's cloned.
                        You'll use this when you evaluate the samples in your sim.

        Returns:
            torch.Tensor: Optimal action, shape (batch_size, action_dim).
        """
        # Get the optimal weights for the first step
        optimal_weights = self.get_optimal_action_weights()  # (batch_size, num_policy_options)
        # Get the actions from each policy
        actions = []
        for p in self.policies_callables:
            actions.append(p(tensordict).pop("action"))  # (batch_size, action_dim)
        actions = torch.stack(actions, dim=-1)  # (num_policy_options, batch_size, action_dim)
        # Compute the weighted sum of actions
        action = (actions * optimal_weights.unsqueeze(1)).sum(dim=-1)  # (batch_size, action_dim)
        return action

    def _get_rollout_weights(self):
        """
        Performs the sampling step of MPPI.
        Generates `num_samples` sequences of policy-blending weights.

        Args:
            tensordict: The current observation/state. Not directly used for sampling
                        logic here but is standard for planners. It's cloned.
                        You'll use this when you evaluate the samples in your sim.

        Returns:
            torch.Tensor: Sampled policy weight sequences, shape
                          (num_samples, rollout_steps, num_policy_options).
                          These are the weights you'll use to linearly combine your policies.
        """
        # 1. Shift mean logits for receding horizon control
        # The mean logits from the previous timestep are shifted, and the last step is re-initialized.
        shifted_logits = self.mean_logits[1:].clone()
        # Initialize the new last step logits to zeros (or another strategy)
        new_last_step_logits = torch.zeros(1, self.env.n_robots, self.num_policy_options, device=self.device)
        self.mean_logits = torch.cat((shifted_logits, new_last_step_logits), dim=0)
        logit_perturbations = (
            torch.randn(self.num_samples, self.rollout_steps, self.env.n_robots, self.num_policy_options, device=self.device) * self.noise_sigma
        )
        # Store perturbations for the update_mean_logits step
        self.last_logit_perturbations = logit_perturbations
        # 3. Create sampled logits
        # Broadcast mean_logits: (1, rollout_steps, 3) + (num_samples, rollout_steps, 3)
        sampled_logits = self.mean_logits.unsqueeze(0) + self.last_logit_perturbations
        # 4. Convert sampled logits to policy weights using softmax
        # This ensures weights for the 3 policies sum to 1 and are non-negative at each step.
        # sampled_policy_weights shape: (num_samples, rollout_steps, num_policy_options)
        sampled_policy_weights = F.softmax(sampled_logits / self.temperature, dim=-1)
        # These sampled_policy_weights are what you'll use in your environment
        # to simulate the `num_samples` different trajectories.
        return sampled_policy_weights

    def update_mean_logits(self, costs: torch.Tensor):
        """
        Updates mean logits using costs for each batch element independently.

        Args:
            costs (torch.Tensor): Costs for each sample and batch element,
                                shape (num_samples, batch_size).
        """
        if self.last_logit_perturbations is None:
            raise RuntimeError("`__call__` must be invoked first.")
        if costs.shape != (self.num_samples, self.env.n_robots):
            raise ValueError(f"Expected costs shape ({self.num_samples}, {self.env.n_robots}), got {costs.shape}")
        costs = costs.to(self.device)

        # Compute exponent terms per sample and batch element
        exponent_terms = -costs / self.lambda_  # (num_samples, batch_size)
        max_exponents = torch.max(exponent_terms, dim=0)[0]  # (batch_size,)
        stable_exponent_terms = exponent_terms - max_exponents.unsqueeze(0)  # (num_samples, batch_size)
        importance_weights = torch.exp(stable_exponent_terms)  # (num_samples, batch_size)
        # Normalize weights per batch element
        sum_of_weights = torch.sum(importance_weights, dim=0)  # (batch_size,)
        epsilon = 1e-9
        normalized_weights = importance_weights / (sum_of_weights.unsqueeze(0) + epsilon)  # (num_samples, batch_size)
        # Compute update delta
        weighted_perturbations = (
            normalized_weights.unsqueeze(1).unsqueeze(-1) * self.last_logit_perturbations
        )  # (num_samples, 1, batch_size, 1) * (num_samples, rollout_steps, batch_size, num_policy_options)
        update_delta = torch.sum(weighted_perturbations, dim=0)  # (rollout_steps, batch_size, num_policy_options)
        self.mean_logits += update_delta
        self.last_logit_perturbations = None

    def get_optimal_action_weights(self):
        """
        Returns the policy blending weights for the *first step* of the current
        optimized mean trajectory for each batch element.

        Returns:
            torch.Tensor: Optimal policy blending weights, shape
                          (batch_size, num_policy_options).
        """
        # Get the mean logits for the first step of the planned horizon
        first_step_mean_logits = self.mean_logits[0, :, :]  # (batch_size, num_policy_options)
        # Convert these logits to policy weights using softmax with temperature
        optimal_weights = F.softmax(first_step_mean_logits / self.temperature, dim=-1)
        return optimal_weights


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
    planner = MultiPolicyMPPIPlanner(
        actor_value_wrapper=actor_value_wrapper,
        vecnorm_obj=vecnorm,
        weights=p_sds,
        vecnorm_weights=vecnorm_sds,
        env=env,
        device=device,
        rng=rng,
        rollout_steps=steps,
        num_samples=num_samples,
        lambda_=lambda_,
        noise_sigma=noise_sigma,
        temperature=temperature,
    )
    env.reset()
    rollout = env.rollout(config.max_eval_steps, planner, break_when_all_done=True, break_when_any_done=False)
    log = log_from_eval_rollout(rollout)
    return log


configs = ["file_natural_cracks.yaml", "file_natural.yaml", "file_ridge.yaml", "file_sine.yaml"]


def mppi_eval():
    FULL_RESULT_DIR = RESULTS_DIR / "ensembled_mppi"
    FULL_RESULT_DIR.mkdir(parents=True, exist_ok=True)
    for cfg in configs:
        test_config = OmegaConf.load(CONFIGS_DIR / cfg)
        train_config = PPOExperimentConfig(**test_config)
        train_config.num_robots = NUM_ENVS_PER_EVAL
        train_config.max_eval_steps = MAX_EVAL_STEPS
        train_config.objective_opts["cache_size"] = 1000
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
                    log = get_eval_rollout(train_config)
            results.append(log)
        savename = cfg.split(".")[0]
        with open(FULL_RESULT_DIR / f"{savename}.pkl", "wb") as f:
            pickle.dump(results, f)


if __name__ == "__main__":
    mppi_eval()
