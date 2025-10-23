"""
Minari dataset loader for offline RL data integration.
"""

import minari
from typing import Optional


class MinariDatasetLoader:
    """
    Minari dataset loader class for offline RL data integration.
    """

    def __init__(self):
        """Initialize the Minari dataset loader."""
        pass

    def load_dataset(
        self,
        env_name: str,
        dataset_type: str = "expert",
        max_episodes: Optional[int] = None,
    ):
        """
        Load a Minari dataset based on environment name.

        Args:
            env_name: Environment name (e.g., 'PongNoFrameskip-v4')
            dataset_type: Type of dataset to search for (default: 'expert')
            max_episodes: Maximum episodes to load (None for all)

        Returns:
            Loaded Minari dataset

        Raises:
            ValueError: If no matching dataset is found
        """
        # Get all available datasets
        all_datasets = minari.list_remote_datasets()

        # Extract base environment name
        base_name = env_name.lower()

        # Remove common suffixes
        base_name = (
            base_name.replace("noframeskip", "")
            .replace("-v4", "")
            .replace("-v0", "")
            .replace("-v1", "")
        )
        base_name = base_name.replace("_", "").replace("-", "")

        print(
            f"Searching for datasets matching environment: {env_name} (base: {base_name})"
        )

        # Search for matching datasets
        matching_datasets = []
        for dataset in all_datasets:
            dataset_lower = dataset.lower().replace("_", "").replace("-", "")

            # Check if base name is in dataset name and dataset type matches
            if base_name in dataset_lower and dataset_type.lower() in dataset_lower:
                matching_datasets.append(dataset)

        if not matching_datasets:
            # Try without dataset_type requirement
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

        # Return the first match (prefer expert datasets)
        expert_datasets = [d for d in matching_datasets if "expert" in d.lower()]
        if expert_datasets:
            dataset_name = expert_datasets[0]
        else:
            dataset_name = matching_datasets[0]

        print(f"Found matching dataset: {dataset_name}")

        # Load the dataset
        print(f"Loading Minari dataset: {dataset_name}")
        dataset = minari.load_dataset(dataset_name, download=True)

        if max_episodes:
            print(f"Limiting to {max_episodes} episodes")

        print(f"Successfully loaded dataset: {dataset_name}")
        print(f"Total episodes: {dataset.total_episodes}")
        print(f"Total steps: {dataset.total_steps}")

        return dataset
