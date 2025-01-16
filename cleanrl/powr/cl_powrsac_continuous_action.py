# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/sac/#sac_continuous_actionpy
import os
import random
import time
from dataclasses import dataclass
from functools import wraps

import gymnasium as gym
import numpy as np
import scipy.stats
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.type_aliases import ReplayBufferSamples
from torch.utils.tensorboard import SummaryWriter
from math import ceil, pi, sqrt
from typing import Optional, Union

import logging
from contrastive_repr import SupConLoss

from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

def setup_logging(run_name):
    os.makedirs("logs", exist_ok=True)

    # Clear previous handlers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(f"logs/{run_name}.log"), logging.StreamHandler()],
    )


def log_time(func):
    """Decorator to time and log function execution."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        logging.info(f"{func.__name__} took {elapsed:.4f} seconds")
        return result

    return wrapper

@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
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
    n_envs: int = 16
    """number of parallel environments"""
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
    cont_batch_size: int = 1024
    """the batch size of sample from the reply memory"""
    learning_starts: int = int(1000) #int(1e4)
    """timestep to start learning"""
    policy_lr: float = 3e-4
    """the learning rate of the policy network optimizer"""
    q_lr: float = 3e-4
    """the learning rate of the Q network network optimizer"""
    cont_lr: float = 0.05
    """the learning rate of the contrastive learning network optimizer"""
    policy_frequency: int = 2
    """the frequency of training policy (delayed)"""
    critic_frequency: int = 4
    """the frequency of training critic (delayed)"""
    target_network_frequency: int = 1  # Denis Yarats' implementation delays this by 2.
    """the frequency of updates for the target nerworks"""
    alpha: float = 0.2
    """Entropy regularization coefficient."""
    autotune: bool = True
    """automatic tuning of the entropy coefficient"""
    num_rfs: int = 4096
    """the number of random features"""
    rf_seed: int = 0
    """the seed for the random features"""
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

    
class ReprReplayBuffer(ReplayBuffer):
    def sample_contrastive(self, batch_size, env=None):
        d_size = batch_size // self.n_envs
        # crude sampling.
        assert d_size * 2 < self.size(), "Not enough data to sample from"
        batch_idx = np.random.choice(self.size()//2, d_size, replace=False) * 2

        #no memory optimization done yet
        obs = self.observations[batch_idx].reshape(-1, *self.observations.shape[2:])
        next_obs = self.next_observations[batch_idx].reshape(-1, *self.next_observations.shape[2:])

        data = (
            self._normalize_obs(obs, env),
            self.actions[batch_idx].reshape(-1, *self.actions.shape[2:]),
            self._normalize_obs(next_obs),
            # Only use dones that are not due to timeouts
            # deactivated by default (timeouts is initialized as an array of False)
            
            # not needing reward or done info yet for contrastive learning
            (self.dones[batch_idx] * (1 - self.timeouts[batch_idx])).reshape(-1, 1),
            self._normalize_reward(self.rewards[batch_idx].reshape(-1, 1), env),
        )
        return ReplayBufferSamples(*tuple(map(self.to_torch, data)))




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
            "action_scale",
            torch.tensor(
                (env.single_action_space.high - env.single_action_space.low) / 2.0,
                dtype=torch.float32,
            ),
        )
        self.register_buffer(
            "action_bias",
            torch.tensor(
                (env.single_action_space.high + env.single_action_space.low) / 2.0,
                dtype=torch.float32,
            ),
        )

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (
            log_std + 1
        )  # From SpinUp / Denis Yarats

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

    def MC_sample(self, x, num_samples: int = 10):
        mean, log_std = self(x)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.sample(
            (num_samples,)
        )  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        actions = y_t * self.action_scale + self.action_bias
        return actions

class ContrastRepr(nn.Module):
    def __init__(self, env):
        super().__init__()
        s_size = np.array(env.single_observation_space.shape).prod()
        a_size = np.array(env.single_action_space.shape).prod()
        assert s_size >= a_size

        if s_size > a_size:
            self.a_embed = nn.Sequential(
                nn.Linear(a_size, s_size, bias=False),
                nn.ReLU(),
                nn.Linear(s_size, s_size, bias=False),
            )
        else:
            self.a_embed = nn.Identity()
        
        self.sa_embed = nn.Sequential(
            nn.Linear(s_size * 2, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
        )

        self.next_obs_layer = nn.Sequential(
            nn.ReLU(),
            nn.Linear(256, 256),
        )

        self.r_out = nn.Linear(256, 1)

    def forward(self, s, a, next = False):

        a = self.a_embed(a)
        x = torch.cat([s, a], dim=-1)
        x = self.sa_embed(x)
        if next:
            x = self.next_obs_layer(x)
        repr = F.normalize(x, p=2, dim=-1)
        r_pred = self.r_out(repr)
        return repr, r_pred
    
# ALGO LOGIC: initialize agent here:
class SoftQNetwork(nn.Module):
    def __init__(self, repr_net : ContrastRepr, latent_dim=256):
        super().__init__()

        self.repr_net = repr_net
        # World model
        self.register_buffer("reward_weights", torch.zeros((latent_dim,)))
        self.register_buffer(
            "transition_weights", torch.zeros((latent_dim, latent_dim))
        )
        self.register_buffer("action_value_weights", torch.zeros((latent_dim,)))

    # def orf_embed(self, states, actions):
    #     x = torch.cat([states, actions], dim=-1)
    #     x = self.orf(x)
    #     return x

    def update_repr_net(self, repr_net):
        self.repr_net = repr_net
        return
    def forward(self, x, a):
        x, _ = self.repr_net(x, a)
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

        # X = self.orf_embed(b_obs, b_actions)
        X, _ = self.repr_net(b_obs, b_actions)
        Y = torch.zeros_like(X)

        actions = actor.MC_sample(b_next_obs)
        # MC mean of the features
        for action in actions:
            # Y += self.orf_embed(b_next_obs, action)
            emb, _ = self.repr_net(b_next_obs, action, next=True)
            Y += emb
        Y /= num_mean_samples

        C0 = covariance(X)
        C1 = covariance(X, Y)
        reward_coeffs = covariance(X, b_rewards)
        return C0, C1, reward_coeffs

    def fit_world_model(self, rb: ReplayBuffer, actor: Actor):
        C0, C1, b = log_time(self.compute_covariances)(rb, actor)
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





def ridge_eval(model, x, y, frac=0.9):
        bsz = x.shape[0]
        train_size = int(bsz * frac)
        train_x = x[:train_size]
        test_x = x[train_size:]
        train_y = y[:train_size]
        test_y = y[train_size:]
        model.fit(train_x, train_y)
        pred_y = model.predict(test_x)
        return mean_squared_error(test_y, pred_y)


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
    envs = gym.vector.SyncVectorEnv([make_env(args.env_id, args.seed+i, 0, args.capture_video, run_name) for i in range(args.n_envs)])
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    max_action = float(envs.single_action_space.high[0])

    actor = Actor(envs).to(device)
    repr_net = ContrastRepr(envs).to(device)
    qf1 = SoftQNetwork(repr_net).to(device)
    # qf2 = SoftQNetwork(envs, ((args.rf_seed + 1))).to(device) 
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.policy_lr)

    repr_optimizer = optim.Adam(list(repr_net.parameters()), lr=args.cont_lr)
    contrastive_loss = SupConLoss(temperature=0.1)
    ridge_repr = Ridge(alpha=1e-6)
    ridge_raw = Ridge(alpha=1e-6)


    # Automatic entropy tuning
    if args.autotune:
        target_entropy = -torch.prod(torch.Tensor(envs.single_action_space.shape).to(device)).item()
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha = log_alpha.exp().item()
        a_optimizer = optim.Adam([log_alpha], lr=args.q_lr)
    else:
        alpha = args.alpha

    envs.single_observation_space.dtype = np.float32
    rb = ReprReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        envs.num_envs,
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
        if "final_info" in infos and infos["final_info"][0] is not None:
            for info in infos["final_info"]:
                print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
                break #this is odd, but it's how the original code is written. Single env only maybe
        else:
            if "episode" in infos:
                print(f"global_step={global_step}, episodic_return={info['episode']['r'].mean()}")
                writer.add_scalar("charts/episodic_return", info["episode"]["r"].mean(), global_step)
                writer.add_scalar("charts/episodic_length", info["episode"]["l"].mean(), global_step)
            
            

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
            if global_step % args.critic_frequency == 0:
                # Fitting the world model
                with torch.no_grad():
                    qf1.fit_world_model(rb, actor)
                    # qf2.fit_world_model(rb, actor)
            data = rb.sample(args.batch_size)
           

            if global_step % args.policy_frequency == 0:  # TD 3 Delayed update support
                for _ in range(
                    args.policy_frequency
                ):  # compensate for the delay by doing 'actor_update_interval' instead of 1
                    pi, log_pi, _ = actor.get_action(data.observations)
                    qf1.train()
                    qf1_pi = qf1(data.observations, pi)
                    qf1.eval()
                    # x = torch.cat([data.observations, data.actions], dim=-1)
                    # inputs = qf1.orf.batch_norm(x)
                    # logging.info(f"m: {inputs.mean(0)} | s: {inputs.std(0)}")

                    # qf2.train()
                    # qf2_pi = qf2(data.observations, pi)
                    # qf2.eval()
                    min_qf_pi = qf1_pi #torch.min(qf1_pi, qf2_pi)
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

            
            
            #perform contrastive learning here
            data = rb.sample_contrastive(args.cont_batch_size)
            zero_action = torch.zeros_like(data.actions)
            obs_repr, r_pred = repr_net(data.observations, data.actions)
            with torch.no_grad():  
                next_actions, _ ,_ = actor.get_action(data.next_observations)
            next_obs_repr, _ = repr_net(data.next_observations, next_actions, next=True)

            cont_loss = contrastive_loss(torch.stack([obs_repr, next_obs_repr], dim=1))

            reward_loss = F.mse_loss(r_pred, data.rewards)
            full_loss = cont_loss + reward_loss
            repr_optimizer.zero_grad()
            full_loss.backward()
            repr_optimizer.step()
            qf1.update_repr_net(repr_net)


            # with torch.no_grad():
            #     sa_np = repr_net(data.observations, data.actions).cpu().numpy()
            


            if global_step % 100 == 0:
                writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
                writer.add_scalar("losses/cont_loss", cont_loss.item(), global_step)
                writer.add_scalar("losses/alpha", alpha, global_step)
                print("SPS:", int(global_step / (time.time() - start_time)))
                writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
                if args.autotune:
                    writer.add_scalar("losses/alpha_loss", alpha_loss.item(), global_step)
                

                r_np = data.rewards.cpu().numpy()
                sa_raw = torch.cat([data.observations, data.actions], dim=-1).cpu().numpy()
            # print(sa_np.shape, r_np.shape, sa_raw.shape)
                writer.add_scalar("losses/repr_reward", reward_loss.item(), global_step)
                writer.add_scalar("losses/baseline_reward", ridge_eval(ridge_raw, sa_raw, r_np), global_step)
                

                # with torch.no_grad():
                #     sa_np = repr_net(data.observations, data.actions).cpu().numpy()
                # r_np = data.rewards.cpu().numpy()
                # sa_raw = torch.cat([data.observations, data.actions], dim=-1).cpu().numpy()
                # print(ridge_eval(ridge_repr, sa_np, r_np), ridge_eval(ridge_raw, sa_raw, r_np))

    envs.close()
    writer.close()