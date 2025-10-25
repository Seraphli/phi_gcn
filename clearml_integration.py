"""
ClearML integration module for experiment tracking and result uploading
"""

import os
from typing import Dict, Any, Optional
import torch
import numpy as np

try:
    from clearml import Task, Logger

    CLEARML_AVAILABLE = True
except ImportError:
    CLEARML_AVAILABLE = False
    print(
        "Warning: ClearML not installed. Run 'pip install clearml' to enable experiment tracking."
    )


class ClearMLTracker:
    """ClearML experiment tracker"""

    def __init__(self, project_name: str = "Phi-GCN", task_name: str = None):
        """
        Initialize ClearML tracker

        Args:
            project_name: Project name
            task_name: Task name
        """
        self.task = None
        self.logger = None

        if not CLEARML_AVAILABLE:
            print("ClearML not available. Tracking disabled.")
            return

        try:
            # Create ClearML task, force creating new task instead of reusing existing one
            self.task = Task.init(
                project_name=project_name,
                task_name=task_name,
                reuse_last_task_id=False,  # Force creating new task
            )

            # Get logger
            self.logger = self.task.get_logger()

            print(f"ClearML task initialized: {project_name}/{task_name}")

        except Exception as e:
            print(f"Failed to initialize ClearML: {e}")
            self.task = None
            self.logger = None

    def log_hyperparameters(self, args):
        """Log hyperparameters"""
        if not self.task:
            return

        try:
            # Convert args to dictionary
            if hasattr(args, "__dict__"):
                params = vars(args)
            else:
                params = args

            self.task.connect(params)
            print("Hyperparameters logged to ClearML")

        except Exception as e:
            print(f"Failed to log hyperparameters: {e}")

    def log_scalar(self, title: str, series: str, value: float, iteration: int):
        """Log scalar value"""
        if not self.logger:
            return

        try:
            self.logger.report_scalar(
                title=title, series=series, value=value, iteration=iteration
            )
        except Exception as e:
            print(f"Failed to log scalar {title}/{series}: {e}")

    def log_metrics(self, metrics: Dict[str, float], iteration: int, prefix: str = ""):
        """Batch log metrics"""
        if not self.logger:
            return

        for key, value in metrics.items():
            title = f"{prefix}/{key}" if prefix else key
            self.log_scalar(title, "value", value, iteration)

    def log_model(self, model: torch.nn.Module, model_name: str = "model"):
        """Save and upload model"""
        if not self.task:
            return

        try:
            # Save model to temporary file
            model_path = f"{model_name}.pth"
            torch.save(model.state_dict(), model_path)

            # Upload model file
            self.task.upload_artifact(model_name, artifact_object=model_path)

            # Clean up temporary file
            if os.path.exists(model_path):
                os.remove(model_path)

            print(f"Model {model_name} uploaded to ClearML")

        except Exception as e:
            print(f"Failed to upload model: {e}")

    def log_episode_rewards(self, rewards, timesteps: int):
        """Log episode rewards"""
        if not rewards or not self.logger:
            return

        try:
            # Convert to list for compatibility
            rewards_list = list(rewards)

            # Log recent reward
            recent_reward = rewards_list[-1]
            self.log_scalar(
                "Training", "Latest Episode Reward", recent_reward, timesteps
            )

            # Log average reward (last 100 episodes)
            if len(rewards_list) >= 100:
                avg_reward = np.mean(rewards_list[-100:])
                self.log_scalar(
                    "Training", "Average Reward (100 episodes)", avg_reward, timesteps
                )
            elif len(rewards_list) > 0:
                avg_reward = np.mean(rewards_list)
                self.log_scalar(
                    "Training", "Average Reward (all episodes)", avg_reward, timesteps
                )

            # Log maximum reward
            max_reward = max(rewards_list)
            self.log_scalar("Training", "Max Reward", max_reward, timesteps)

        except Exception as e:
            print(f"Failed to log episode rewards: {e}")

    def log_training_progress(
        self,
        total_frames: int,
        fps: float,
        value_loss: float,
        action_loss: float,
        entropy_loss: float,
        iteration: int,
    ):
        """Log training progress"""
        if not self.logger:
            return

        try:
            self.log_scalar("Progress", "Total Frames", total_frames, iteration)
            self.log_scalar("Progress", "FPS", fps, iteration)
            self.log_scalar("Losses", "Value Loss", value_loss, iteration)
            self.log_scalar("Losses", "Action Loss", action_loss, iteration)
            self.log_scalar("Losses", "Entropy Loss", entropy_loss, iteration)

        except Exception as e:
            print(f"Failed to log training progress: {e}")

    def log_gcn_metrics(
        self, gcn_alpha: float, reward_propagation_stats: Dict, iteration: int
    ):
        """Log GCN related metrics"""
        if not self.logger:
            return

        try:
            self.log_scalar("GCN", "Alpha", gcn_alpha, iteration)

            for key, value in reward_propagation_stats.items():
                self.log_scalar("GCN", key, value, iteration)

        except Exception as e:
            print(f"Failed to log GCN metrics: {e}")

    def log_environment_info(
        self, env_name: str, num_processes: int, num_steps: int, num_frames: int
    ):
        """Log environment information"""
        if not self.task:
            return

        try:
            env_config = {
                "environment": env_name,
                "num_processes": num_processes,
                "num_steps": num_steps,
                "total_frames": num_frames,
            }

            self.task.connect(env_config, name="Environment Config")
            print("Environment info logged to ClearML")

        except Exception as e:
            print(f"Failed to log environment info: {e}")

    def finish(self):
        """Finish task"""
        if self.task:
            try:
                self.task.close()
                print("ClearML task completed")
            except Exception as e:
                print(f"Failed to close ClearML task: {e}")


def create_clearml_tracker(args) -> ClearMLTracker:
    """
    Convenience function to create ClearML tracker

    Args:
        args: Training parameters

    Returns:
        ClearMLTracker instance
    """
    task_name = getattr(args, "clearml_task")
    if task_name is None or task_name == "null":
        # Auto-generate task name
        env_name = getattr(args, "env_name")
        algo = getattr(args, "algo")
        seed = getattr(args, "seed")
        task_name = f"{env_name}_{algo}_seed{seed}"

    # Use project name from config
    project_name = getattr(args, "clearml_project")

    # Create tracker
    tracker = ClearMLTracker(project_name=project_name, task_name=task_name)

    # Log hyperparameters
    tracker.log_hyperparameters(args)

    return tracker
