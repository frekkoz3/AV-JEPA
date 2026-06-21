from typing import Dict, Any, Tuple
import copy

import torch
from numpy import floating

from src.policy.policy import *
from src.policy.algorithms import *
from src.policy.epsilon import *
from src.policy.losses import *

from src.game.snake import *
from src.utils.utils import *


class PolicyPPO(Policy):
    """Subclass for PPO-based policies"""

    def __init__(self,  **kwargs):

        assert kwargs.get("network", "ConvPPO") in ["ConvPPO", "AttentionPPO"], f"Invalid network type: {kwargs.get('network')}. Must be 'ConvPPO' or 'AttentionPPO'."
        assert kwargs.get("loss", "PPOLoss") in ["PPOLoss", "A2CLoss"], f"Invalid loss type: {kwargs.get('loss')}. Must be 'PPOLoss' or 'A2CLoss'."
        assert kwargs.get("epsilon_strategy", "EpsilonConstant") == "EpsilonConstant", f"Invalid epsilon strategy: {kwargs.get('epsilon_strategy')}. Must be 'EpsilonConstant' for PolicyPPO."

        super().__init__(**kwargs)

        self.n_inner_epochs = kwargs.get("n_inner_epochs", 4)


    def _format_state(self, state : Any) -> torch.Tensor:
        """
        Formats the input state into a suitable format for the PPO network:
                [batch_size, n_channels, raw_pixels] = [1, 1, 400]
        """
        state_tensor = state if isinstance(state, torch.Tensor) else torch.tensor(state, dtype=torch.float32, device=self.device)

        if state_tensor.shape == (20, 20):
            state_tensor = state_tensor.flatten().unsqueeze(0)

        return state_tensor


    @torch.no_grad()
    def get_action(self,
                   state : torch.Tensor | Tuple[torch.Tensor, ...],
                   greedy = False) -> Tuple[torch.Tensor | Any, tuple[Any, Any]]:
        """Selects an action based on the current state using a policy derived from the PPO algorithm."""
        dist, value = self.network(state)
        if greedy:
            action = torch.argmax(dist.logits, dim=-1)
        else:
            action = dist.sample().cpu().numpy()
        return action, (dist.log_prob(torch.tensor(action)).cpu().numpy(), value.squeeze(-1))


    def train(self):
        """PPO full training loop"""
        assert self.environment is not None, "Environment must be set for training"
        pass


    def update_parameters(self, trajectory : torch.Tensor | Tuple[torch.Tensor, ...]) -> Dict[str, floating[Any] | Any]:
        """
        Updates the parameters of the architecture based on policy gradient optimization.

        Parameters
        ----------
        trajectory : torch.Tensor | Tuple[torch.Tensor, ...]
            A trajectory of states, actions, rewards, and other relevant information used for computing the policy gradient update.

        Returns
        -------
        loss : Dict[str,  float]
            A dictionary containing the computed losses for monitoring and analysis.

        Notes
        -----
        The update is slower than the one for DQN, since it requires (in e2e_jepa.update_parameters() ) to compute the full trajectory of states, actions, next states, rewards, dones.
        """
        z_states, _, z_next_states, actions, rewards, dones, log_probs, values = trajectory

        # Compute Advantages and Returns using GAE
        returns, advantages = self.loss.compute_advantages(last_state_value = values[-1], rewards = rewards, values = values, dones = dones)

        loss_history = []
        dist_history = []

        self.network.train()
        for _ in range(self.n_inner_epochs):
            dist, value = self.network(z_states)
            loss = self.loss(dist = dist, values = value, actions = actions, returns = returns, advantages = advantages, old_log_probs = log_probs)

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=0.5)
            self.optimizer.step()

            loss_history.append(loss.item())
            dist_history.append(dist.probs.mean(dim=0).cpu().etach().numpy())

        return {"loss": np.mean(loss_history), "mean_distribution": np.mean(dist_history, axis=0)}



class PolicyDQN(Policy):
    """Subclass for DQN-based policies"""

    def __init__(self,**kwargs):

        assert kwargs.get("network", "ConvDQN") in ["ConvDQN", "AttentionDQN", "DQN"], f"Invalid network type: {kwargs.get('network')}. Must be 'ConvDQN' or 'AttentionDQN' or 'DQN'."
        assert kwargs.get("loss", "MSELoss") == "MSELoss", \
            f"Invalid loss type: {kwargs.get('loss')}. Must be 'MSELoss'."
        assert kwargs.get("epsilon_strategy", "EpsilonConstant") in ["EpsilonConstant", "EpsilonGreedy"], f"Invalid epsilon strategy: {kwargs.get('epsilon_strategy')}. Must be 'EpsilonConstant' or 'EpsilonGreedy'."

        super().__init__(**kwargs)

        self.target_network = copy.deepcopy(self.network)
        self.target_net_update_freq = kwargs.get("target_update_freq", 25)
        self.epoch = 0

        self.replay_buffer = deque(maxlen=kwargs.get("replay_buffer_size", 10000))
        self.batch_size = kwargs.get("batch_size", 64)



    @torch.no_grad()
    def get_action(self,
                   state : torch.Tensor | Tuple[torch.Tensor, ...],
                   greedy : bool = False) -> Tuple[torch.Tensor | Any, tuple[Any, Any]]:
        """Selects an action based on the current state using an epsilon-greedy strategy."""
        if not greedy and np.random.rand() < self.epsilon_strategy.eps:
            action = np.random.randint(0, self.network.output.out_features)
        else:
            q_values = self.network(state)
            action = torch.argmax(q_values, dim=-1).cpu().numpy()
        return action, (None, None)


    def train(self):
        """DQN full training loop"""
        pass


    def _format_state(self, state : Any) -> torch.Tensor:
            """Formats the input state into a suitable format for the PPO network."""
        pass


    def update_parameters(self,
                          init_state : torch.Tensor | Tuple[torch.Tensor, ...],
                          next_state : torch.Tensor | Tuple[torch.Tensor],
                          rewards : torch.Tensor,
                          dones : torch.Tensor,
                          ) -> Dict[str, float | int]:
        """
        Computes a TD learning step for the DQN architecture.

        Parameters
        ----------
        init_state : torch.Tensor | Tuple[torch.Tensor, ...]
            The initial state or a tuple of states from which to compute the DQN update.
        next_state : torch.Tensor | Tuple[torch.Tensor]
            The next state or a tuple of next states corresponding to the initial states.
        rewards : torch.Tensor
            The rewards received after taking actions in the initial states.
        dones : torch.Tensor
            A tensor indicating whether the episodes have terminated after taking actions in the initial states.

        Returns
        -------
        loss : Dict[str,  float]
            A dictionary containing the computed losses for monitoring and analysis.
        """
        # Compute Q-Values for the initial state
        q_values = self.network(init_state)
        online_q_values = q_values.gather(1, torch.argmax(q_values, dim=-1, keepdim=True)).squeeze(-1)

        # Compute Target Q-Values for the next state using the target network
        with torch.no_grad():
            next_q_values = self.target_network(next_state)
            max_next_q_values, _ = torch.max(next_q_values, dim=-1)
            target_q_values = rewards + self.reward_discount * max_next_q_values * (1 - dones)

        # Compute the loss (MSE) between online and target Q-Values
        loss = self.loss(online_q_values, target_q_values)

        # Optimize parameters
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.epsilon_strategy.step()

        # Update target network
        self.epoch += 1
        if self.epoch % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.network.state_dict())

        return {"loss": loss.item()}
