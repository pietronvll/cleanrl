# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/sac/#sac_continuous_actionpy
import os
import random
import time
from math import ceil, pi, sqrt
from dataclasses import dataclass
from typing import Optional, Union

import gymnasium as gym
import numpy as np
import torch
import scipy.stats
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions.normal import Normal
import tyro
from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    rf_seed: int = 0
    """the seed for the random features"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    # Algorithm specific arguments
    env_id: str = "Hopper-v4"
    """the environment id of the task"""
    total_timesteps: int = 1000000
    """total timesteps of the experiments"""
    buffer_size: int = int(1e6)
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 0.005
    """target smoothing coefficient (default: 0.005)"""
    batch_size: int = 256
    """the batch size of sample from the reply memory"""
    learning_starts: int = 5e3
    """timestep to start learning"""
    policy_lr: float = 3e-4
    """the learning rate of the policy network optimizer"""
    q_lr: float = 1e-3
    """the learning rate of the Q network network optimizer"""
    policy_frequency: int = 2
    """the frequency of training policy (delayed)"""
    target_network_frequency: int = 1  # Denis Yarats' implementation delays this by 2.
    """the frequency of updates for the target nerworks"""
    alpha: float = 0.2
    """Entropy regularization coefficient."""
    autotune: bool = True
    """automatic tuning of the entropy coefficient"""
    num_rfs: int = 4096
    """the number of random features"""
    reg: float = 1e-5
    """regularization fop the POWR world model"""


def make_env(env_id, seed, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)
        return env

    return thunk

def covariance(
    X: torch.Tensor,
    Y: Optional[torch.Tensor] = None,
    center: bool = False,
    weights: Optional[Union[float, torch.Tensor]] = None,
):
    """Covariance matrix

    Args:
        X (torch.Tensor): Input covariates of shape ``(samples, features)``.
        Y (Optional[torch.Tensor], optional): Output covariates of shape ``(samples, features)`` Defaults to None.
        center (bool, optional): Whether to compute centered covariances. Defaults to True.

    Returns:
        torch.Tensor: Covariance matrix of shape ``(features, features)``. If ``Y is not None`` computes the cross-covariance between X and Y.
    """
    assert X.ndim == 2
    if weights is None:
        weights = torch.rsqrt(torch.tensor(X.shape[0]))
    else:
        weights = torch.sqrt(torch.tensor(weights))

    if weights.ndim == 1:
        assert weights.shape[0] == X.shape[0]
        weights = weights.unsqueeze(1)
    elif weights.ndim > 1:
        raise ValueError("Weights should be a scalar or 1D vector of shape (samples,)")
    else:
        pass

    if Y is None:
        _X = weights * X
        if center:
            _X = _X - _X.mean(dim=0, keepdim=True)
        return torch.mm(_X.T, _X)
    else:
        assert Y.ndim == 2
        _X = weights * X
        _Y = weights * Y
        if center:
            _X = _X - _X.mean(dim=0, keepdim=True)
            _Y = _Y - _Y.mean(dim=0, keepdim=True)
        return torch.mm(_X.T, _Y)
    
class OrthogonalRandomFeatures(torch.nn.Module):
    def __init__(
        self,
        # Possibly just pass the environment specs here.
        input_dim: int,
        num_random_features: int = 1024,
        length_scale: Union[float, torch.Tensor] = 1.0,
        rng_seed: Optional[int] = None,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.num_random_features = num_random_features
        if torch.is_tensor(length_scale):
            assert length_scale.ndim == 0 or (
                length_scale.ndim == 1 and length_scale.shape[0] == self.state_dim
            )
        else:
            length_scale = torch.tensor(length_scale)

        W = self._sample_orf_matrix(self.input_dim, self.num_random_features, rng_seed)
        self.register_buffer("rff_matrix", W)
        self.register_buffer(
            "random_offset", torch.rand(self.num_random_features) * 2 * pi
        )
        self.register_buffer("length_scale", length_scale)
        self.layer_norm = nn.LayerNorm(
            self.input_dim, bias=False, elementwise_affine=False
        )

    def _sample_orf_matrix(self, input_dim, num_random_features, rng_seed):
        num_folds = ceil(num_random_features / input_dim)
        rng_torch = rng_seed if rng_seed is None else torch.manual_seed(rng_seed)

        G = torch.randn(
            num_folds,
            input_dim,
            input_dim,
            generator=rng_torch,
        )
        Q, _ = torch.linalg.qr(
            G, mode="complete"
        )  # The _columns_ in each batch of matrices in Q are orthonormal.

        Q = Q.transpose(
            -1, -2
        )  # The _rows_ in each batch of matrices in Q are orthonormal.

        S = torch.tensor(
            scipy.stats.chi.rvs(
                input_dim,
                size=(num_folds, input_dim, 1),
                random_state=rng_seed,
            ),
        )

        W = Q * S  # [num_folds, input_dim, input_dim]
        W = torch.cat(
            [W[fold_idx, ...] for fold_idx in range(num_folds)], dim=0
        )  # Concatenate columns [num_folds*input_dim, input_dim]
        W = W[:num_random_features, :]
        return W.T

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        inputs = self.layer_norm(inputs)
        Z = torch.tensordot(
            inputs
            / self.length_scale,  # Length_scale should broadcast to the batch dimension
            self.rff_matrix.to(inputs.dtype),
            dims=1,
        )
        assert self.random_offset.shape[0] == Z.shape[-1]
        Z = torch.cos(Z + self.random_offset) / sqrt(0.5 * self.num_random_features)
        return Z  # [batch_size, num_random_features]

LOG_STD_MAX = 2
LOG_STD_MIN = -5


class Actor(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(np.array(env.single_observation_space.shape).prod(), 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mean = nn.Linear(256, np.prod(env.single_action_space.shape))
        self.fc_logstd = nn.Linear(256, np.prod(env.single_action_space.shape))
        # action rescaling
        self.register_buffer(
            "action_scale", torch.tensor((env.action_space.high - env.action_space.low) / 2.0, dtype=torch.float32)
        )
        self.register_buffer(
            "action_bias", torch.tensor((env.action_space.high + env.action_space.low) / 2.0, dtype=torch.float32)
        )

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)  # From SpinUp / Denis Yarats

        return mean, log_std

    def get_action(self, x):
        mean, log_std = self(x)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean

# ALGO LOGIC: initialize agent here:
class SoftQNetwork(nn.Module):
    def __init__(self, env, rf_seed = 0):
        super().__init__()
        state_dim = np.prod(env.observation_space.shape)
        action_dim = np.prod(env.action_space.shape)
        self.orf = OrthogonalRandomFeatures(
            state_dim + action_dim,
            num_random_features=args.num_rfs,
            length_scale=sqrt(state_dim + action_dim),
            rng_seed=rf_seed,
        )
        # World model
        latent_dim = args.num_rfs
        self.register_buffer("reward_weights", torch.zeros((latent_dim,)))
        self.register_buffer(
            "transition_weights", torch.zeros((latent_dim, latent_dim))
        )
        self.register_buffer("action_value_weights", torch.zeros((latent_dim,)))

    def orf_embed(self, states, actions):
        x = torch.cat([states, actions], dim=-1)
        x = self.orf(x)
        return x
    
    def forward(self, x, a):
        x = self.orf_embed(x, a)
        action_value_fn = torch.einsum("...d, d -> ...", x, self.action_value_weights)
        return action_value_fn

    def compute_covariances(self, rb: ReplayBuffer, actor: Actor, num_mean_samples=10):
        obs = rb.observations if rb.full else rb.observations[: rb.pos]
        next_obs = rb.next_observations if rb.full else rb.next_observations[: rb.pos]
        rewards = rb.rewards if rb.full else rb.rewards[: rb.pos]
        actions = rb.actions if rb.full else rb.actions[: rb.pos]

        b_obs = rb.to_torch(rb.swap_and_flatten(obs)).float()
        b_next_obs = rb.to_torch(rb.swap_and_flatten(next_obs)).float()
        b_rewards = rb.to_torch(rb.swap_and_flatten(rewards)).float()
        b_actions = rb.to_torch(rb.swap_and_flatten(actions)).float()
        X = self.orf_embed(b_obs, b_actions)
        Y = torch.zeros_like(X)

        # MC mean of the features
        for _ in range(num_mean_samples):
            action, _, _ = actor.get_action(b_next_obs)
            Y += self.orf_embed(b_next_obs, action)
        Y /= num_mean_samples

        C0 = covariance(X)
        C1 = covariance(X, Y)
        reward_coeffs = covariance(X, b_rewards)
        return C0, C1, reward_coeffs
    
    def fit_world_model(self, rb: ReplayBuffer, actor: Actor):
        C0, C1, b = self.compute_covariances(rb, actor)
        C0.diagonal().add_(args.reg)  # Add regularization
        C0_chol = torch.linalg.cholesky(C0)
        T_ridge = torch.cholesky_solve(C1, C0_chol)
        self.transition_weights = T_ridge.clone()
        reward_KRR = torch.cholesky_solve(b.view(-1, 1), C0_chol)
        self.reward_weights.copy_(reward_KRR.clone()[:, 0])
        # Inplace evaluation of Id - discount*T_ridge
        T_ridge *= -args.gamma
        T_ridge.diagonal().add_(1)
        sol = torch.linalg.solve(T_ridge, reward_KRR)
        self.action_value_weights.copy_(sol[:, 0])


if __name__ == "__main__":
    import stable_baselines3 as sb3

    if sb3.__version__ < "2.0":
        raise ValueError(
            """Ongoing migration: run the following command to install the new dependencies:
poetry run pip install "stable_baselines3==2.0.0a1"
"""
        )

    args = tyro.cli(Args)
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = gym.vector.SyncVectorEnv([make_env(args.env_id, args.seed, 0, args.capture_video, run_name)])
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    max_action = float(envs.single_action_space.high[0])

    actor = Actor(envs).to(device)
    qf1 = SoftQNetwork(envs, args.rf_seed).to(device)
    qf2 = SoftQNetwork(envs, (args.rf_seed+1)).to(device)
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.policy_lr)

    # Automatic entropy tuning
    if args.autotune:
        target_entropy = -torch.prod(torch.Tensor(envs.single_action_space.shape).to(device)).item()
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha = log_alpha.exp().item()
        a_optimizer = optim.Adam([log_alpha], lr=args.q_lr)
    else:
        alpha = args.alpha

    envs.single_observation_space.dtype = np.float32
    rb = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        handle_timeout_termination=False,
    )
    start_time = time.time()

    # TRY NOT TO MODIFY: start the game
    obs, _ = envs.reset(seed=args.seed)
    for global_step in range(args.total_timesteps):
        # ALGO LOGIC: put action logic here
        if global_step < args.learning_starts:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            actions, _, _ = actor.get_action(torch.Tensor(obs).to(device))
            actions = actions.detach().cpu().numpy()

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if "final_info" in infos:
            for info in infos["final_info"]:
                print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
                break

        # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc:
                real_next_obs[idx] = infos["final_observation"][idx]
        rb.add(obs, real_next_obs, actions, rewards, terminations, infos)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # ALGO LOGIC: training.
        if global_step > args.learning_starts:
            data = rb.sample(args.batch_size)

            # Fitting the world model
            with torch.no_grad():
                qf1.fit_world_model(rb, actor)
                qf2.fit_world_model(rb, actor)

            if global_step % args.policy_frequency == 0:  # TD 3 Delayed update support
                for _ in range(
                    args.policy_frequency
                ):  # compensate for the delay by doing 'actor_update_interval' instead of 1
                    pi, log_pi, _ = actor.get_action(data.observations)
                    qf1_pi = qf1(data.observations, pi)
                    qf2_pi = qf2(data.observations, pi)
                    min_qf_pi = torch.min(qf1_pi, qf2_pi)
                    actor_loss = ((alpha * log_pi) - min_qf_pi).mean()

                    actor_optimizer.zero_grad()
                    actor_loss.backward()
                    actor_optimizer.step()

                    if args.autotune:
                        with torch.no_grad():
                            _, log_pi, _ = actor.get_action(data.observations)
                        alpha_loss = (-log_alpha.exp() * (log_pi + target_entropy)).mean()

                        a_optimizer.zero_grad()
                        alpha_loss.backward()
                        a_optimizer.step()
                        alpha = log_alpha.exp().item()

            if global_step % 100 == 0:
                writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
                writer.add_scalar("losses/alpha", alpha, global_step)
                print("SPS:", int(global_step / (time.time() - start_time)))
                writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
                if args.autotune:
                    writer.add_scalar("losses/alpha_loss", alpha_loss.item(), global_step)

    envs.close()
    writer.close()