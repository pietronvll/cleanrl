# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/sac/#sac_continuous_actionpy
import os
import copy
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.type_aliases import ReplayBufferSamples
from torch.utils.tensorboard import SummaryWriter

from contrastive_repr import SupConLoss, SpectralConLoss, NoiseConLoss

from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

from datetime import datetime
import socket

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
    total_timesteps: int = 600_000
    """total timesteps of the experiments"""
    buffer_size: int = int(1e6)
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 0.005
    """target smoothing coefficient (default: 0.005)"""
    feature_tau: float = 0.005
    """target smoothing coefficient (default: 0.005)"""
    batch_size: int = 256
    """the batch size of sample from the replay memory"""
    cont_batch_size: int = 1024
    """the batch size of sample from the replay memory"""
    learning_starts: int = int(1000) #int(1e4)
    """timestep to start learning"""
    policy_lr: float = 3e-4
    """the learning rate of the policy network optimizer"""
    q_lr: float = 3e-4
    """the learning rate of the Q network network optimizer"""
    feat_lr: float = 1e-4
    """the learning rate of the contrastive learning network optimizer"""
    policy_frequency: int = 2
    """the frequency of training policy (delayed)"""
    target_network_frequency: int = 1  # Denis Yarats' implementation delays this by 2.
    """the frequency of updates for the target nerworks"""
    critic_layers: int = 1
    """the number of layers in the Q networks"""
    critic_hidden_dim: int = 256
    """the hidden dimension of the critic networks"""
    rep_loss: str = "nce" # "supervised" or "spectral" or "nce"
    """the loss function for the representation learning"""
    extra_feature_steps: int = 3
    """the number of extra feature steps to train Mu and Phi"""
    use_feature_target: bool = False
    """whether to use feature target""" # NOT yet understood
    feature_dim: int = 256
    """the dimension of the feature"""
    feat_hidden_dim: int = 256  
    """the hidden dimension of the neural networks"""
    freeze_feature: bool = True
    """whether to freeze the feature learning during the training of the policy"""
    reward_prediction_loss: bool = True
    """whether to use reward prediction loss"""
    reward_weight: float = 0.5
    """the weight of the reward prediction loss"""
    alpha: float = 0
    """Entropy regularization coefficient."""
    autotune: bool = False
    """automatic tuning of the entropy coefficient"""


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

class Phi(nn.Module):
    """
    phi: s, a -> z_phi in R^d
    """
    def __init__(
        self, 
        state_dim,
        action_dim,
        feature_dim=1024,
        hidden_dim=1024,
        ):

        super(Phi, self).__init__()

        self.l1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, feature_dim)

    def forward(self, state, action):
        x = torch.cat([state, action], axis=-1)
        z = F.elu(self.l1(x)) 
        z = F.elu(self.l2(z)) 
        z_phi = self.l3(z)
        z_phi = F.normalize(z_phi, p=2, dim=-1)
        return z_phi

class Mu(nn.Module):
    """
    mu': s' -> z_mu in R^d
    """
    def __init__(
        self, 
        state_dim,
        feature_dim=1024,
        hidden_dim=1024,
        ):

        super(Mu, self).__init__()

        self.l1 = nn.Linear(state_dim , hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, feature_dim)

    def forward(self, state):
        z = F.elu(self.l1(state))
        z = F.elu(self.l2(z))
        # bounded mu's output
        z_mu = F.tanh(self.l3(z)) 
        z_mu = F.normalize(z_mu, p=2, dim=-1)
        # z_mu = self.l3(z)
        return z_mu
     
class Theta(nn.Module):
    """
    Linear theta 
    <phi(s, a), theta> = r 
    """
    def __init__(
        self, 
        feature_dim=1024,
        ):

        super(Theta, self).__init__()

        self.l = nn.Linear(feature_dim, 1)

    def forward(self, feature):
        r = self.l(feature)
        return r
     
# class ContrastRepr(nn.Module):
#     def __init__(self, env,  feature_dim: int = 256, hidden_dim: int  = 256):
#         super().__init__()
#         s_size = np.array(env.single_observation_space.shape).prod()
#         a_size = np.array(env.single_action_space.shape).prod()

#         self.phi = Phi(s_size, a_size, feature_dim, hidden_dim)
#         self.mu = Mu(s_size, feature_dim)

#         self.theta = Theta(feature_dim) # for reward predictions

#     def forward(self, state, action = None, next_state=False):
#         if next_state:
#             return self.mu(state), None
#         else:
#             x = self.phi(state, action)
#             reward_prediction = self.theta(x)
#             return x, reward_prediction
       

# ALGO LOGIC: initialize agent here:
class SoftQNetwork(nn.Module):
    def __init__(self, feature_dim = 256, n_layers=1, hidden_dim=256):
        super().__init__()
        # self.embedder = embedder

        # create Q network layers depending on the number of layers and hidden_dim using a loop
        if n_layers < 1:
            raise ValueError("n_layers must be greater than 1")
        elif n_layers == 1:
            self.critic_layers = nn.ModuleList([nn.Linear(feature_dim, 1)])
        else:
            self.critic_layers = nn.ModuleList()
            self.critic_layers.append(nn.Linear(feature_dim, hidden_dim))
            for _ in range(n_layers - 2):
                self.critic_layers.append(nn.ReLU())   
                self.critic_layers.append(nn.Linear(hidden_dim, hidden_dim)) 
            self.critic_layers.append(nn.ReLU())
            self.critic_layers.append(nn.Linear(hidden_dim, 1))  

    def forward(self, z):
        x = z
        for layer in self.critic_layers:
            x = layer(x)
        return x


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
            "action_scale", torch.tensor((env.single_action_space.high - env.single_action_space.low) / 2.0, dtype=torch.float32)
        )
        self.register_buffer(
            "action_bias", torch.tensor((env.single_action_space.high + env.single_action_space.low) / 2.0, dtype=torch.float32)
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

def get_run_name(args, current_date=None):
    if current_date is None:
        current_date = datetime.today().strftime("%Y_%m_%d_%H_%M_%S")
    return (
        str(current_date)
        + "_"
        + str(args.env_id)
        + "_"
        + str(args.rep_loss) + "_sac_continuous_action"
        + "_clayers="
        + str(args.critic_layers)
        + "_rp_loss="
        + str(args.reward_prediction_loss)
        + "_fdim="
        + str(args.feature_dim)
        + "_rw="
        + str(args.reward_weight)
        + "_seed"
        + str(args.seed)
        + "_"
        + socket.gethostname()
    )


if __name__ == "__main__":
    import stable_baselines3 as sb3

    if sb3.__version__ < "2.0":
        raise ValueError(
            """Ongoing migration: run the following command to install the new dependencies:
poetry run pip install "stable_baselines3==2.0.0a1"
"""
        )

    args = tyro.cli(Args)
    run_name = get_run_name(args) #f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
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
    feature_dim = args.feature_dim
    state_dim = np.array(envs.single_observation_space.shape).prod()
    action_dim = np.array(envs.single_action_space.shape).prod()
    # Representation learning Nets
    # phi(s, a) -> z_phi
    phi= Phi(state_dim, action_dim, feature_dim, args.feat_hidden_dim).to(device)
    frozen_phi = copy.deepcopy(phi)
    if args.use_feature_target:
        phi_target = copy.deepcopy(phi)
        frozen_phi_target = copy.deepcopy(phi)
    # mu(s') -> z_mu
    mu = Mu(state_dim, feature_dim, args.feat_hidden_dim).to(device)

    #<phi(s, a), theta> = r 
    theta = Theta(feature_dim).to(device)

    qf1 = SoftQNetwork( feature_dim, args.critic_layers, args.critic_hidden_dim).to(device)
    qf2 = SoftQNetwork( feature_dim, args.critic_layers, args.critic_hidden_dim).to(device)
    qf1_target = SoftQNetwork( feature_dim, args.critic_layers, args.critic_hidden_dim).to(device)
    qf2_target = SoftQNetwork( feature_dim, args.critic_layers, args.critic_hidden_dim).to(device)
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())
    q_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=args.q_lr)
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.policy_lr)
    feature_optimizer = torch.optim.Adam(
            list(phi.parameters()) + list(mu.parameters()) + list(theta.parameters()),
            lr=args.feat_lr)

    if args.rep_loss == "supervised":
        contrastive_loss = SupConLoss(temperature=0.1)
    elif args.rep_loss == "spectral":
        contrastive_loss = SpectralConLoss()
    elif args.rep_loss == "nce":
        contrastive_loss = NoiseConLoss(device)
    else:
        raise ValueError("Unknown representation learning loss: ", args.rep_loss)
    
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

            #perform contrastive learning here
            for _ in range(args.extra_feature_steps+1):
                data = rb.sample_contrastive(args.cont_batch_size)
                z_phi = phi(data.observations, data.actions)
                z_mu_next = mu(data.next_observations)
                r_pred= theta(z_phi)

                cont_loss = contrastive_loss(z_phi, z_mu_next)
                r_prediction_loss = F.mse_loss(r_pred, data.rewards).mean()
                feature_loss = cont_loss  + args.reward_weight*r_prediction_loss*int(args.reward_prediction_loss)
                feature_optimizer.zero_grad()
                feature_loss.backward()
                feature_optimizer.step()
                
                # Update the feature network if needed
                if args.use_feature_target:
                    for param, target_param in zip(phi.parameters(), phi_target.parameters()):
                        target_param.data.copy_(args.feature_tau * param.data + (1 - args.feature_tau) * target_param.data)
  
            # copy phi to frozen phi
            frozen_phi.load_state_dict(phi.state_dict().copy())
            if args.use_feature_target:
                frozen_phi_target.load_state_dict(phi.state_dict().copy())

            data = rb.sample(args.batch_size)
            with torch.no_grad():
                next_state_actions, next_state_log_pi, _ = actor.get_action(data.next_observations)
                if args.use_feature_target:
                    z_phi = frozen_phi_target(data.observations, data.actions)
                    z_phi_next = frozen_phi_target(data.next_observations, next_state_actions)
                else:
                    z_phi = frozen_phi(data.observations, data.actions)
                    z_phi_next = frozen_phi(data.next_observations, next_state_actions)

                qf1_next_target = qf1_target(z_phi_next)
                qf2_next_target = qf2_target(z_phi_next)
                min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - alpha * next_state_log_pi
                next_q_value = data.rewards.flatten() + (1 - data.dones.flatten()) * args.gamma * (min_qf_next_target).view(-1)

            qf1_a_values = qf1(z_phi).view(-1)
            qf2_a_values = qf2(z_phi).view(-1)
            qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
            qf2_loss = F.mse_loss(qf2_a_values, next_q_value)

            qf_loss = qf1_loss + qf2_loss 

            # optimize the model
            q_optimizer.zero_grad()
            qf_loss.backward()
            q_optimizer.step()

            if global_step % args.policy_frequency == 0:  # TD 3 Delayed update support
                for _ in range(
                    args.policy_frequency
                ):  # compensate for the delay by doing 'actor_update_interval' instead of 1
                    pi, log_pi, _ = actor.get_action(data.observations)
                    if args.freeze_feature:
                        z_phi = frozen_phi(data.observations, pi)
                    else:
                        z_phi = phi(data.observations, pi)
                    qf1_pi = qf1(z_phi)
                    qf2_pi = qf2(z_phi)
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

            # update the target networks
            if global_step % args.target_network_frequency == 0:
                for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                for param, target_param in zip(qf2.parameters(), qf2_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
            


            if global_step % 100 == 0:
                writer.add_scalar("losses/qf1_values", qf1_a_values.mean().item(), global_step)
                writer.add_scalar("losses/qf2_values", qf2_a_values.mean().item(), global_step)
                writer.add_scalar("losses/qf1_loss", qf1_loss.item(), global_step)
                writer.add_scalar("losses/qf2_loss", qf2_loss.item(), global_step)
                writer.add_scalar("losses/qf_loss", qf_loss.item() / 2.0, global_step)
                writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
                writer.add_scalar("losses/cont_error", cont_loss.item(), global_step)
                writer.add_scalar("losses/r_pred_error", r_prediction_loss.item(), global_step)
                writer.add_scalar("losses/feature_loss", feature_loss.item(), global_step)
                writer.add_scalar("losses/alpha", alpha, global_step)
                print("SPS:", int(global_step / (time.time() - start_time)))
                writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
                if args.autotune:
                    writer.add_scalar("losses/alpha_loss", alpha_loss.item(), global_step)
                

                r_np = data.rewards.cpu().numpy()
                sa_raw = torch.cat([data.observations, data.actions], dim=-1).cpu().numpy()
            # print(sa_np.shape, r_np.shape, sa_raw.shape)
                # writer.add_scalar("losses/baseline_reward", ridge_eval(ridge_raw, sa_raw, r_np), global_step)
                

                # with torch.no_grad():
                #     sa_np = embedder(data.observations, data.actions).cpu().numpy()
                # r_np = data.rewards.cpu().numpy()
                # sa_raw = torch.cat([data.observations, data.actions], dim=-1).cpu().numpy()
                # print(ridge_eval(ridge_repr, sa_np, r_np), ridge_eval(ridge_raw, sa_raw, r_np))

    envs.close()
    writer.close()