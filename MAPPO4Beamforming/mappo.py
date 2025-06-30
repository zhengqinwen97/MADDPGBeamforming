import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal


# === Utility Layers ===
class AddConstant(nn.Module):
    def __init__(self, value):
        super().__init__()
        self.value = value

    def forward(self, x):
        return x + self.value


# === Actor-Critic Network ===
class ActorCritic(nn.Module):
    """
    CTDE-style MAPPO Actor-Critic Network.
    Uses centralized critic and decentralized actor.
    """
    def __init__(self, obs_dim, act_dim, total_power, hidden_dim=64, centralized=True, n_agents=1):
        super().__init__()
        self.total_power = total_power
        self.centralized = centralized
        critic_input_dim = obs_dim * n_agents if centralized else obs_dim

        # === Actor Mean ===
        self.actor_mean_net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, act_dim),
            AddConstant(1 / (2 * act_dim))
        )

        # === Actor Std ===
        self.actor_std_net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, act_dim)
        )

        # === Centralized Critic ===
        self.critic = nn.Sequential(
            nn.Linear(critic_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def get_action_distribution(self, obs):
        """
        Returns action distribution parameters (mean, std)
        obs: [N, obs_dim]
        """
        mean = self.actor_mean_net(obs)
        std = self.actor_std_net(obs)
        std = torch.clamp(std, min=1e-2, max=0.5)
        return mean, std

    def evaluate(self, critic_input):
        """
        Evaluates centralized critic value.
        critic_input: [N, obs_dim*n_agents] if centralized else [N, obs_dim]
        """
        return self.critic(critic_input).squeeze(-1)  # return shape: [N]

class MAPPO:
    def __init__(self, n_agents, obs_dim, beam_count, args):
        self.n_agents = n_agents
        self.obs_dim = obs_dim
        self.act_dim = beam_count
        self.gamma = args['gamma']
        self.lam = args['gae_lambda']
        self.clip_ratio = args['clip_ratio']
        self.policy_coef = args['policy_coef']
        self.value_coef = args['value_coef']
        self.entropy_coef = args['entropy_coef']
        self.max_grad_norm = args['max_grad_norm']
        self.total_power = args['total_power']
        self.device = args['device']

        # --- Policy network ---
        self.policy = ActorCritic(
            obs_dim=obs_dim,
            act_dim=self.act_dim,
            total_power=self.total_power,
            centralized=True,
            n_agents=n_agents
        ).to(self.device)

        self.optimizer = optim.Adam(self.policy.parameters(), lr=args['lr'])
        self.lr_scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.95)

    def act(self, obs):
        mean, std = self.policy.get_action_distribution(obs)
        dist = Normal(mean, std)

        z = dist.rsample()
        bounded = torch.tanh(z)
        proportions = (bounded + 1.0) / 2.0
        actions = proportions * self.total_power

        # Log prob correction for tanh squashing
        log_prob = dist.log_prob(z) - torch.log(1 - bounded.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1)

        return actions, log_prob

    def evaluate_value(self, centralized_obs):
        if centralized_obs.ndim == 2:  # [N, obs_dim]
            centralized_obs = centralized_obs.reshape(1, -1)
            return self.policy.evaluate(centralized_obs).squeeze(-1)
        elif centralized_obs.ndim == 3:  # [T, N, obs_dim]
            T, N, obs_dim = centralized_obs.shape
            centralized_obs = centralized_obs.reshape(T, N * obs_dim)
            values = self.policy.evaluate(centralized_obs)  # [T]
            return values.unsqueeze(1).expand(-1, N)         # [T, N]

    def compute_loss(self, obs, all_obs, act, adv, ret, old_logp):
        # Normalize actions for log_prob calculation
        proportions = act / self.total_power
        epsilon = 0.05
        proportions = (1 - epsilon) * proportions + epsilon / proportions.shape[-1]

        # Normalize advantages
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        # Actor forward
        mean, std = self.policy.get_action_distribution(obs)
        dist = Normal(mean, std)
        log_probs = dist.log_prob(proportions).sum(dim=-1)

        # PPO objective
        ratio = torch.exp(log_probs - old_logp)
        surrogate1 = ratio * adv
        surrogate2 = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * adv
        policy_loss = -torch.mean(torch.min(surrogate1, surrogate2))

        # Critic loss
        value = self.evaluate_value(all_obs)
        value_loss = torch.mean((ret - value) ** 2)

        # Entropy bonus
        entropy = dist.entropy().sum(dim=-1)
        mean_entropy = entropy.mean()

        # Total loss
        total_loss = (
            self.policy_coef * policy_loss +
            self.value_coef * value_loss -
            self.entropy_coef * mean_entropy
        )

        return total_loss, policy_loss, value_loss, mean_entropy

    def update(self, total_loss):
        self.optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        self.optimizer.step()

        self.lr_scheduler.step()
        for param_group in self.optimizer.param_groups:
            if param_group['lr'] < 1e-5:
                param_group['lr'] = 1e-5