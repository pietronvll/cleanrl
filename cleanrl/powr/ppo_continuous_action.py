# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_continuous_actionpy
import logging
import os
import random
import time
from dataclasses import dataclass
from math import ceil, pi, sqrt
from typing import Optional, Union

import gymnasium as gym
import numpy as np
import scipy.stats
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from stable_baselines3.common.buffers import ReplayBuffer
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter

# Saving videos from an headless server
os.environ["MUJOCO_GL"] = "egl"


# Usage
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
    wandb_entity: Optional[str] = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    save_model: bool = False
    """whether to save model into the `runs/{run_name}` folder"""
    upload_model: bool = False
    """whether to upload the saved model to huggingface"""
    hf_entity: str = ""
    """the user or org name of the model repository from the Hugging Face Hub"""

    # Algorithm specific arguments
    env_id: str = "HalfCheetah-v4"
    """the id of the environment"""
    total_timesteps: int = 1000000
    """total timesteps of the experiments"""
    learning_rate: float = 3e-4
    """the learning rate of the optimizer"""
    num_envs: int = 1
    """the number of parallel game environments"""
    num_steps: int = 2048
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 32
    """the number of mini-batches"""
    update_epochs: int = 10
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.0
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: Optional[float] = None
    """the target KL divergence threshold"""
    num_rfs: int = 2048
    """the number of random features"""
    rf_seed: int = 0
    """the seed for the random features"""
    reg: float = 1e-4
    """regularization fop the POWR world model"""
    buffer_size: int = 2**15
    """size of the replay buffer"""
    dump_buffer: bool = False
    """saving the replay buffer to disk"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""


def make_env(env_id, idx, capture_video, run_name, gamma):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(
                env, f"videos/{run_name}", step_trigger=lambda step: step % 50000 == 0
            )
        else:
            env = gym.make(env_id)
        env = gym.wrappers.FlattenObservation(
            env
        )  # deal with dm_control's Dict observation space
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.ClipAction(env)
        env = gym.wrappers.NormalizeObservation(env)
        env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
        env = gym.wrappers.NormalizeReward(env, gamma=gamma)
        env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
        return env

    return thunk


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


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


class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        state_dim = np.prod(envs.single_observation_space.shape)
        action_dim = np.prod(envs.single_action_space.shape)
        self.orf = OrthogonalRandomFeatures(
            state_dim + action_dim,
            num_random_features=args.num_rfs,
            length_scale=sqrt(state_dim + action_dim),
            rng_seed=args.rf_seed,
        )
        # World model
        latent_dim = args.num_rfs
        self.register_buffer("reward_weights", torch.zeros((latent_dim,)))
        self.register_buffer(
            "transition_weights", torch.zeros((latent_dim, latent_dim))
        )
        self.register_buffer("action_value_weights", torch.zeros((latent_dim,)))

        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(state_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, action_dim), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, action_dim))

    def orf_embed(self, states, actions):
        x = torch.cat([states, actions], dim=-1)
        x = self.orf(x)
        return x

    def get_reward(self, states, actions):
        x = self.orf_embed(states, actions)
        reward_fn = torch.einsum("...d, d -> ...", x, self.reward_weights)
        return reward_fn

    def compute_covariances(self, rb: ReplayBuffer, num_mean_samples=10):
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
        action_mean = self.actor_mean(b_obs)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        for _ in range(num_mean_samples):
            Y += self.orf_embed(b_next_obs, probs.sample())
        Y /= num_mean_samples

        C0 = covariance(X)
        C1 = covariance(X, Y)
        reward_coeffs = covariance(X, b_rewards)
        return C0, C1, reward_coeffs

    def fit_world_model(self, rb: ReplayBuffer):
        C0, C1, b = self.compute_covariances(rb)
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

    def get_action_value(self, states, actions):
        x = self.orf_embed(states, actions)
        action_value_fn = torch.einsum("...d, d -> ...", x, self.action_value_weights)
        return action_value_fn

    def get_value(self, state, num_mean_samples: int = 1000):
        # Sample actions from the policy and compute the mean empirically
        action_mean = self.actor_mean(state)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        actions = probs.sample(
            (num_mean_samples,)
        )  # [num_mean_samples, num_states, act_dim]
        broadcasted_state = state.expand(num_mean_samples, actions.shape[1], -1)
        x = self.orf_embed(broadcasted_state, actions)
        value_fn = torch.einsum("...d, d -> ...", x, self.action_value_weights)
        value_fn = value_fn.mean(0, keepdim=True)
        return value_fn

    def get_action_and_value(self, x, action=None):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return (
            action,
            probs.log_prob(action).sum(1),
            probs.entropy().sum(1),
            self.get_value(x),
        )


if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    setup_logging(run_name)
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
        "|param|value|\n|-|-|\n%s"
        % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [
            make_env(args.env_id, i, args.capture_video, run_name, args.gamma)
            for i in range(args.num_envs)
        ]
    )
    assert isinstance(
        envs.single_action_space, gym.spaces.Box
    ), "only continuous action space is supported"

    agent = Agent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # ALGO Logic: Storage setup
    obs = torch.zeros(
        (args.num_steps, args.num_envs) + envs.single_observation_space.shape
    ).to(device)
    actions = torch.zeros(
        (args.num_steps, args.num_envs) + envs.single_action_space.shape
    ).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    rb = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        handle_timeout_termination=False,
    )

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)

    for iteration in range(1, args.num_iterations + 1):
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        # Batch norm statistics
        agent.train()
        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminations, truncations, infos = envs.step(
                action.cpu().numpy()
            )
            next_done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = (
                torch.Tensor(next_obs).to(device),
                torch.Tensor(next_done).to(device),
            )

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        logging.info(
                            f"global_step={global_step}, episodic_return={info['episode']['r']}"
                        )
                        writer.add_scalar(
                            "charts/episodic_return", info["episode"]["r"], global_step
                        )
                        writer.add_scalar(
                            "charts/episodic_length", info["episode"]["l"], global_step
                        )

            # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
            real_next_obs = next_obs.clone()
            for idx, trunc in enumerate(truncations):
                if trunc:
                    real_next_obs[idx] = torch.Tensor(
                        infos["final_observation"][idx]
                    ).to(device)
            rb.add(
                obs[step].numpy(force=True),
                real_next_obs.numpy(force=True),
                action.numpy(force=True),
                reward,
                terminations,
                infos,
            )

        agent.eval()
        # bootstrap value if not done
        # Generalized Advantage Estimation
        gae_start = time.time()
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = (
                    rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                )
                advantages[t] = lastgaelam = (
                    delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                )
            returns = advantages + values
        logging.info(f"GAE estimation took {time.time() - gae_start:.3f}")
        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)
        b_rewards = rewards.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        wm_fit_time = 0
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]
                _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    b_obs[mb_inds], b_actions[mb_inds]
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [
                        ((ratio - 1.0).abs() > args.clip_coef).float().mean().item()
                    ]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                        mb_advantages.std() + 1e-8
                    )

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(
                    ratio, 1 - args.clip_coef, 1 + args.clip_coef
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()
            wm_fit_start = time.time()
            with torch.no_grad():
                agent.fit_world_model(rb)
            wm_fit_time += time.time() - wm_fit_start

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        if args.dump_buffer:
            np.savez(
                f"logs/dump_{run_name}",
                observations=rb.observations,
                next_observations=rb.next_observations,
                actions=rb.actions,
                rewards=rb.rewards,
                dones=rb.dones,
                timeouts=rb.timeouts,
            )

        logging.info(f"world model fit time {wm_fit_time/args.update_epochs:3f}")
        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # Reward fitting:
        with torch.no_grad():
            r_pred, r_true = (
                agent.get_reward(b_obs, b_actions).numpy(force=True),
                b_rewards.numpy(force=True),
            )
            var_r = np.var(r_true)
            explained_reward_var = (
                np.nan if var_r == 0 else 1 - np.var(r_true - r_pred) / var_r
            )
            logging.info(f"explained reward variance {explained_reward_var:.2f}")
        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar(
            "charts/learning_rate", optimizer.param_groups[0]["lr"], global_step
        )
        # writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance_value", explained_var, global_step)
        writer.add_scalar(
            "losses/explained_variance_reward", explained_reward_var, global_step
        )
        steps_per_sec = int(global_step / (time.time() - start_time))
        logging.info(f"steps per second:{steps_per_sec}")
        writer.add_scalar("charts/SPS", steps_per_sec, global_step)

    if args.save_model:
        model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
        torch.save(agent.state_dict(), model_path)
        print(f"model saved to {model_path}")
        from cleanrl_utils.evals.ppo_eval import evaluate

        episodic_returns = evaluate(
            model_path,
            make_env,
            args.env_id,
            eval_episodes=10,
            run_name=f"{run_name}-eval",
            Model=Agent,
            device=device,
            gamma=args.gamma,
        )
        for idx, episodic_return in enumerate(episodic_returns):
            writer.add_scalar("eval/episodic_return", episodic_return, idx)

        if args.upload_model:
            from cleanrl_utils.huggingface import push_to_hub

            repo_name = f"{args.env_id}-{args.exp_name}-seed{args.seed}"
            repo_id = f"{args.hf_entity}/{repo_name}" if args.hf_entity else repo_name
            push_to_hub(
                args,
                episodic_returns,
                repo_id,
                "PPO",
                f"runs/{run_name}",
                f"videos/{run_name}-eval",
            )

    envs.close()
    writer.close()
