import minari
import hashlib
import numpy as np
from typing import Optional
import gymnasium as gym
from gymnasium import spaces
import networkx as nx
import torch


def wrap_minari_replay(env):
    from atari_wrappers import WarpFrame, FrameStack, ClipRewardEnv
    from envs import TransposeImage

    env = WarpFrame(env)
    env = ClipRewardEnv(env)
    env = FrameStack(env, 4)
    env = TransposeImage(env)
    return env


class ReplayEnv(gym.Env):
    def __init__(self, episode_data):
        self.episode_data = episode_data
        self.step_idx = 0
        self.max_steps = len(episode_data.observations) - 1

        obs_sample = episode_data.observations[0]
        if isinstance(obs_sample, np.ndarray):
            self.observation_space = spaces.Box(
                low=0, high=255, shape=obs_sample.shape, dtype=obs_sample.dtype
            )
        else:
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32
            )

        action_sample = episode_data.actions[0]
        if isinstance(action_sample, np.ndarray):
            if action_sample.shape == ():
                self.action_space = spaces.Discrete(int(action_sample) + 1)
            else:
                self.action_space = spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=action_sample.shape,
                    dtype=action_sample.dtype,
                )
        else:
            self.action_space = spaces.Discrete(2)

    def reset(self, **kwargs):
        self.step_idx = 0
        return self.episode_data.observations[0], {}

    def step(self, action):
        if self.step_idx >= self.max_steps:
            return self.episode_data.observations[-1], 0.0, True, False, {}

        obs = self.episode_data.observations[self.step_idx + 1]
        reward = self.episode_data.rewards[self.step_idx]
        terminated = (
            self.episode_data.terminations[self.step_idx]
            if hasattr(self.episode_data, "terminations")
            else False
        )
        truncated = (
            self.episode_data.truncations[self.step_idx]
            if hasattr(self.episode_data, "truncations")
            else False
        )

        self.step_idx += 1
        return obs, reward, terminated, truncated, {}


class MinariDatasetLoader:
    def __init__(self):
        self.dataset = None
        self.dataset_name = None
        self.original_env_name = None
        self.states_value = {}
        self.state_access_count = 0
        self.expected_state_shape = None
        self._shape_verified = False
        self.offline_episodes = []  # List of episodes, each episode contains [observations, rewards]

    def load_dataset(
        self,
        env_name: str,
        dataset_type: str = "expert",
        max_episodes: Optional[int] = None,
    ):
        self.original_env_name = env_name
        all_datasets = minari.list_remote_datasets()

        base_name = env_name.lower()
        base_name = (
            base_name.replace("noframeskip", "")
            .replace("-v4", "")
            .replace("-v0", "")
            .replace("-v1", "")
        )
        base_name = base_name.replace("_", "").replace("-", "")

        print(f"[Minari] Searching datasets for: {env_name}")

        matching_datasets = []
        for dataset in all_datasets:
            dataset_lower = dataset.lower().replace("_", "").replace("-", "")
            if base_name in dataset_lower and dataset_type.lower() in dataset_lower:
                matching_datasets.append(dataset)

        if not matching_datasets:
            for dataset in all_datasets:
                dataset_lower = dataset.lower().replace("_", "").replace("-", "")
                if base_name in dataset_lower:
                    matching_datasets.append(dataset)

        if not matching_datasets:
            available_datasets = [
                d
                for d in all_datasets
                if any(keyword in d.lower() for keyword in ["atari", "mujoco", "d4rl"])
            ][:10]
            raise ValueError(
                f"No dataset found for environment '{env_name}'. "
                f"Available datasets (first 10): {available_datasets}"
            )

        expert_datasets = [d for d in matching_datasets if "expert" in d.lower()]
        dataset_name = expert_datasets[0] if expert_datasets else matching_datasets[0]

        print(f"[Minari] Loading dataset: {dataset_name}")
        dataset = minari.load_dataset(dataset_name, download=True)

        if max_episodes:
            print(f"[Minari] Limiting to {max_episodes} episodes")

        self.dataset = dataset
        self.dataset_name = dataset_name
        print(
            f"[Minari] Loaded {dataset.total_episodes} episodes, {dataset.total_steps} steps"
        )
        return dataset

    def _compute_state_md5(self, state):
        return hashlib.md5(state.tobytes()).hexdigest()

    def _create_wrapped_env(self, episode_data):
        env = ReplayEnv(episode_data)
        is_atari = "NoFrameskip" in self.original_env_name or hasattr(
            env.unwrapped, "ale"
        )
        if is_atari:
            env = wrap_minari_replay(env)
        return env

    def compute_state_values(self, gamma=0.99, device=None):
        total_states_processed = 0
        episode_count = 0
        self.offline_episodes = []

        for episode_data in self.dataset.iterate_episodes():
            episode_count += 1
            wrapped_env = self._create_wrapped_env(episode_data)
            obs, _ = wrapped_env.reset()

            if obs.shape != (4, 84, 84):
                raise ValueError(
                    f"Wrapped environment obs shape mismatch! Expected: (4, 84, 84), Got: {obs.shape}"
                )

            if self.expected_state_shape is None:
                self.expected_state_shape = obs.shape

            episode_observations = [obs]
            episode_rewards = []

            for step_idx in range(len(episode_data.rewards)):
                reward = episode_data.rewards[step_idx]
                episode_rewards.append(reward)

                if step_idx < len(episode_data.actions):
                    action = episode_data.actions[step_idx]
                    obs, _, terminated, truncated, _ = wrapped_env.step(action)
                    episode_observations.append(obs)

                    if terminated or truncated:
                        break

            returns = self._calculate_returns(episode_rewards, gamma)

            for obs, return_value in zip(episode_observations[:-1], returns):
                state_md5 = self._compute_state_md5(obs)
                if state_md5 in self.states_value:
                    self.states_value[state_md5] = max(
                        self.states_value[state_md5], return_value
                    )
                else:
                    self.states_value[state_md5] = return_value
                total_states_processed += 1

            # Store each episode separately for graph construction
            # Convert observations to torch tensors, stack and move to device here
            obs_tensors = [
                torch.from_numpy(obs).float() for obs in episode_observations[:-1]
            ]
            if len(obs_tensors) > 0:
                stacked_obs = torch.stack(obs_tensors)
                if device is not None:
                    stacked_obs = stacked_obs.to(device)
            else:
                stacked_obs = torch.empty(0)
            self.offline_episodes.append(
                {"observations": stacked_obs, "rewards": episode_rewards}
            )

        total_observations = sum(
            len(ep["observations"]) for ep in self.offline_episodes
        )
        print(
            f"[Minari] Episodes: {self.dataset.total_episodes}, States with values: {len(self.states_value)}"
        )
        print(
            f"[Minari] Stored {len(self.offline_episodes)} episodes with {total_observations} observations for graph construction"
        )

    def _calculate_returns(self, rewards, gamma):
        returns = []
        running_return = 0.0

        for reward in reversed(rewards):
            running_return = reward + gamma * running_return
            returns.append(running_return)

        returns.reverse()
        return returns

    def get_state_value(self, state):
        if not self._shape_verified:
            if self.expected_state_shape is None:
                raise ValueError(
                    "Expected state shape not set. Please call compute_state_values() first."
                )

            if state.shape != self.expected_state_shape:
                raise ValueError(
                    f"State shape mismatch! Expected: {self.expected_state_shape}, "
                    f"Got: {state.shape}. Input state must have the same dimensions as "
                    f"the offline dataset states processed during compute_state_values()."
                )

            self._shape_verified = True

        state_md5 = self._compute_state_md5(state)
        if state_md5 in self.states_value:
            self.state_access_count += 1
            return self.states_value[state_md5]
        return 0.0

    def build_graph_data_from_offline(self, actor_critic, device):
        graph_data_list = []

        # Process each episode separately
        for episode_idx, episode in enumerate(self.offline_episodes):
            obs_tensor = episode["observations"]
            episode_rewards = episode["rewards"]

            if obs_tensor.numel() == 0:
                continue

            # Get hidden features from actor_critic network
            with torch.no_grad():
                episode_length = obs_tensor.shape[0]
                dummy_hidden = torch.zeros(
                    episode_length, actor_critic.recurrent_hidden_state_size
                ).to(device)
                dummy_masks = torch.ones(episode_length, 1).to(device)

                _, _, _, _, hidden_features = actor_critic.act(
                    obs_tensor, dummy_hidden, dummy_masks
                )

            # Build graph for this episode
            episode_length = len(hidden_features)
            G = nx.Graph()
            gcn_states = []
            rew_states = []

            # Add nodes and edges
            for i in range(episode_length):
                gcn_states.append(hidden_features[i])
                if i > 0:
                    G.add_edge(i - 1, i)

                # Add reward states (only for non-zero rewards or episode end)
                if episode_rewards[i] != 0.0 or i == episode_length - 1:
                    rew_states.append([i, episode_rewards[i]])

            # Create adjacency matrix
            if len(G.nodes) > 1:
                adj = nx.adjacency_matrix(G)
            else:
                import scipy.sparse as sp

                adj = sp.csr_matrix(np.eye(1, dtype="int64"))

            # Store graph data if we have enough data
            if len(gcn_states) > 1 and len(rew_states) > 0:
                graph_data_list.append(
                    {
                        "features": torch.stack(gcn_states),
                        "adj": adj,
                        "rew_states": rew_states,
                    }
                )

        return graph_data_list
