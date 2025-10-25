"""
Configuration management using dataclasses and YAML.
"""

from dataclasses import dataclass
from typing import Optional
import yaml
import argparse
import torch


@dataclass
class Config:
    """Main configuration class."""

    # Algorithm settings
    algo: str = "ppo"

    # Learning parameters
    lr: float = 7e-4
    eps: float = 1e-5
    alpha: float = 0.99
    gamma: float = 0.99
    use_gae: bool = False
    tau: float = 0.95
    entropy_coef: float = 0.01
    value_loss_coef: float = 0.5
    max_grad_norm: float = 0.5

    # Environment settings
    env_name: str = "PongNoFrameskip-v4"
    seed: int = 1
    num_processes: int = 16
    num_steps: int = 5
    num_frames: int = 10000000
    add_timestep: bool = False

    # PPO specific parameters
    ppo_epoch: int = 4
    num_mini_batch: int = 32
    clip_param: float = 0.2

    # Logging and saving
    log_interval: int = 10
    save_interval: int = 100
    eval_interval: Optional[int] = None
    vis_interval: int = 100
    log_dir: str = "/tmp/gym/"
    save_dir: str = "./saved_models/"

    # Hardware settings
    cuda: bool = True

    # Policy settings
    recurrent_policy: bool = False
    use_linear_lr_decay: bool = False

    # Visualization
    vis: bool = False
    port: int = 8097

    # Logging
    use_logger: bool = True
    folder: str = "results"

    # GCN parameters
    gcn_epochs: int = 1
    gcn_lr: float = 1e-3
    gcn_weight_decay: float = 0e-4
    gcn_hidden: int = 64
    gcn_alpha: float = 0.5
    gcn_lambda: float = 10.0
    reward_freq: int = 1

    # ClearML settings
    use_clearml: bool = False
    clearml_project: str = "Phi-GCN"
    clearml_task: Optional[str] = None

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "Config":
        """Load configuration from YAML file."""
        with open(yaml_path, "r") as f:
            yaml_data = yaml.safe_load(f)

        # Convert YAML data to config object
        config = cls()
        for key, value in yaml_data.items():
            if hasattr(config, key):
                # Handle type conversion
                field_type = type(getattr(config, key))
                try:
                    if field_type == bool and isinstance(value, str):
                        # Handle string boolean values
                        converted_value = value.lower() in ("true", "1", "yes", "on")
                    elif field_type in (int, float, str, bool):
                        converted_value = field_type(value)
                    else:
                        converted_value = value
                    setattr(config, key, converted_value)
                except (ValueError, TypeError):
                    # If conversion fails, skip this field
                    continue

        return config

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "Config":
        """Create config from argparse Namespace"""
        config = cls()
        for key, value in vars(args).items():
            if key != "config":  # Skip the config file argument
                config_key = key.replace("-", "_")
                if hasattr(config, config_key):
                    setattr(config, config_key, value)
        return config

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}


def get_args() -> Config:
    """Parse command-line arguments and create config."""
    parser = argparse.ArgumentParser(description="PPO")

    # Config file
    parser.add_argument(
        "--config", type=str, default=None, help="Path to YAML config file"
    )

    # Algorithm settings
    parser.add_argument(
        "--algo", default="ppo", help="algorithm to use: a2c | ppo | acktr"
    )

    # Learning parameters
    parser.add_argument(
        "--lr", type=float, default=7e-4, help="learning rate (default: 7e-4)"
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=1e-5,
        help="RMSprop optimizer epsilon (default: 1e-5)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.99,
        help="RMSprop optimizer alpha (default: 0.99)",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.99,
        help="discount factor for rewards (default: 0.99)",
    )
    parser.add_argument(
        "--use-gae",
        action="store_true",
        default=False,
        help="use generalized advantage estimation",
    )
    parser.add_argument(
        "--tau", type=float, default=0.95, help="gae lambda parameter (default: 0.95)"
    )
    parser.add_argument(
        "--entropy-coef",
        type=float,
        default=0.01,
        help="entropy term coefficient (default: 0.01)",
    )
    parser.add_argument(
        "--value-loss-coef",
        type=float,
        default=0.5,
        help="value loss coefficient (default: 0.5)",
    )
    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=0.5,
        help="max norm of gradients (default: 0.5)",
    )

    # Environment settings
    parser.add_argument(
        "--env-name", default="PongNoFrameskip-v4", help="environment to train on"
    )
    parser.add_argument("--seed", type=int, default=1, help="random seed")
    parser.add_argument(
        "--num-processes",
        type=int,
        default=16,
        help="how many training CPU processes to use",
    )
    parser.add_argument(
        "--num-steps", type=int, default=5, help="number of forward steps in A2C"
    )
    parser.add_argument(
        "--num-frames", type=int, default=10e6, help="number of frames to train"
    )
    parser.add_argument(
        "--add-timestep",
        action="store_true",
        default=False,
        help="add timestep to observations",
    )

    # PPO specific parameters
    parser.add_argument(
        "--ppo-epoch", type=int, default=4, help="number of ppo epochs (default: 4)"
    )
    parser.add_argument(
        "--num-mini-batch",
        type=int,
        default=32,
        help="number of batches for ppo (default: 32)",
    )
    parser.add_argument(
        "--clip-param",
        type=float,
        default=0.2,
        help="ppo clip parameter (default: 0.2)",
    )

    # Logging and saving
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
        help="log interval, one log per n updates (default: 10)",
    )
    parser.add_argument(
        "--save-interval",
        type=int,
        default=100,
        help="save interval, one save per n updates (default: 100)",
    )
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=None,
        help="eval interval, one eval per n updates (default: None)",
    )
    parser.add_argument(
        "--vis-interval",
        type=int,
        default=100,
        help="vis interval, one log per n updates (default: 100)",
    )
    parser.add_argument(
        "--log-dir",
        default="/tmp/gym/",
        help="directory to save agent logs (default: /tmp/gym)",
    )
    parser.add_argument(
        "--save-dir",
        default="./saved_models/",
        help="directory to save agent logs (default: ./saved_models/)",
    )

    # Hardware settings
    parser.add_argument(
        "--cuda", action="store_true", default=True, help="enables CUDA training"
    )

    # Policy settings
    parser.add_argument(
        "--recurrent-policy",
        action="store_true",
        default=False,
        help="use a recurrent policy",
    )
    parser.add_argument(
        "--use-linear-lr-decay",
        action="store_true",
        default=False,
        help="use a linear schedule on the learning rate",
    )

    # Visualization
    parser.add_argument(
        "--vis", action="store_true", default=False, help="enable visdom visualization"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8097,
        help="port to run the server on (default: 8097)",
    )

    # Logging
    parser.add_argument(
        "--use-logger", action="store_true", default=True, help="use logger"
    )
    parser.add_argument("--folder", default="results", help="folder to save results")

    # GCN parameters
    parser.add_argument(
        "--gcn-epochs", type=int, default=1, help="number of gcn epochs (default: 1)"
    )
    parser.add_argument(
        "--gcn-lr", type=float, default=1e-3, help="gcn learning rate (default: 1e-3)"
    )
    parser.add_argument(
        "--gcn-weight-decay",
        type=float,
        default=0e-4,
        help="gcn weight decay (default: 0e-4)",
    )
    parser.add_argument(
        "--gcn-hidden", type=int, default=64, help="gcn hidden size (default: 64)"
    )
    parser.add_argument(
        "--gcn-alpha", type=float, default=0.5, help="gcn alpha (default: 0.5)"
    )
    parser.add_argument(
        "--gcn-lambda", type=float, default=10.0, help="gcn lambda (default: 10.0)"
    )
    parser.add_argument(
        "--reward-freq", type=int, default=1, help="reward frequency (default: 1)"
    )

    # ClearML settings
    parser.add_argument(
        "--use-clearml", action="store_true", default=False, help="use clearml logging"
    )
    parser.add_argument(
        "--clearml-project", type=str, default="Phi-GCN", help="clearml project name"
    )
    parser.add_argument(
        "--clearml-task", type=str, default=None, help="clearml task name"
    )

    args = parser.parse_args()

    # Start with argparse defaults
    config = Config.from_args(args)

    # Override with config file if provided
    if args.config:
        yaml_config = Config.from_yaml(args.config)

        # Override config with YAML values
        for field_name in config.to_dict().keys():
            if hasattr(yaml_config, field_name):
                yaml_value = getattr(yaml_config, field_name)
                current_value = getattr(config, field_name)
                if yaml_value != current_value:
                    setattr(config, field_name, yaml_value)

    # Finally override with explicit command-line args (non-default values)
    # We need to detect which args were explicitly set vs defaults
    parser_defaults = {}
    for action in parser._actions:
        if action.dest != "help" and hasattr(action, "default"):
            parser_defaults[action.dest] = action.default

    for key, value in vars(args).items():
        if key != "config" and key in parser_defaults:
            # Only override if the value is different from parser default
            if value != parser_defaults[key]:
                config_key = key.replace("-", "_")
                if hasattr(config, config_key):
                    setattr(config, config_key, value)

    # Set CUDA availability
    config.cuda = config.cuda and torch.cuda.is_available()

    return config
