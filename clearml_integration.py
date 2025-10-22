
"""
ClearML集成模块：用于实验跟踪和结果上传
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
    print("Warning: ClearML not installed. Run 'pip install clearml' to enable experiment tracking.")

class ClearMLTracker:
    """ClearML实验跟踪器"""
    
    def __init__(self, project_name: str = "Phi-GCN", task_name: str = None):
        """
        初始化ClearML跟踪器
        
        Args:
            project_name: 项目名称
            task_name: 任务名称
        """
        self.task = None
        self.logger = None
        
        if not CLEARML_AVAILABLE:
            print("ClearML not available. Tracking disabled.")
            return
            
        try:
            # 创建ClearML任务，强制创建新任务而不是重用现有任务
            self.task = Task.init(
                project_name=project_name,
                task_name=task_name,
                reuse_last_task_id=False  # 强制创建新任务
            )
            
            # 获取logger
            self.logger = self.task.get_logger()
            
            print(f"ClearML task initialized: {project_name}/{task_name}")
            
        except Exception as e:
            print(f"Failed to initialize ClearML: {e}")
            self.task = None
            self.logger = None
    
    def log_hyperparameters(self, args):
        """记录超参数"""
        if not self.task:
            return
            
        try:
            # 将args转换为字典
            if hasattr(args, '__dict__'):
                params = vars(args)
            else:
                params = args
                
            self.task.connect(params)
            print("Hyperparameters logged to ClearML")
            
        except Exception as e:
            print(f"Failed to log hyperparameters: {e}")
    
    def log_scalar(self, title: str, series: str, value: float, iteration: int):
        """记录标量值"""
        if not self.logger:
            return
            
        try:
            self.logger.report_scalar(
                title=title,
                series=series,
                value=value,
                iteration=iteration
            )
        except Exception as e:
            print(f"Failed to log scalar {title}/{series}: {e}")
    
    def log_metrics(self, metrics: Dict[str, float], iteration: int, prefix: str = ""):
        """批量记录指标"""
        if not self.logger:
            return
            
        for key, value in metrics.items():
            title = f"{prefix}/{key}" if prefix else key
            self.log_scalar(title, "value", value, iteration)
    
    def log_model(self, model: torch.nn.Module, model_name: str = "model"):
        """保存并上传模型"""
        if not self.task:
            return
            
        try:
            # 保存模型到临时文件
            model_path = f"{model_name}.pth"
            torch.save(model.state_dict(), model_path)
            
            # 上传模型文件
            self.task.upload_artifact(model_name, artifact_object=model_path)
            
            # 清理临时文件
            if os.path.exists(model_path):
                os.remove(model_path)
                
            print(f"Model {model_name} uploaded to ClearML")
            
        except Exception as e:
            print(f"Failed to upload model: {e}")
    
    def log_episode_rewards(self, rewards, timesteps: int):
        """记录episode奖励"""
        if not rewards or not self.logger:
            return
            
        try:
            # 转换为list以确保兼容性
            rewards_list = list(rewards)
            
            # 记录最近的奖励
            recent_reward = rewards_list[-1]
            self.log_scalar("Training", "Latest Episode Reward", recent_reward, timesteps)
            
            # 记录平均奖励（最近100个episode）
            if len(rewards_list) >= 100:
                avg_reward = np.mean(rewards_list[-100:])
                self.log_scalar("Training", "Average Reward (100 episodes)", avg_reward, timesteps)
            elif len(rewards_list) > 0:
                avg_reward = np.mean(rewards_list)
                self.log_scalar("Training", "Average Reward (all episodes)", avg_reward, timesteps)
            
            # 记录最大奖励
            max_reward = max(rewards_list)
            self.log_scalar("Training", "Max Reward", max_reward, timesteps)
            
        except Exception as e:
            print(f"Failed to log episode rewards: {e}")
    
    def log_training_progress(self, total_frames: int, fps: float, 
                            value_loss: float, action_loss: float, 
                            entropy_loss: float, iteration: int):
        """记录训练进度"""
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
    
    def log_gcn_metrics(self, gcn_alpha: float, reward_propagation_stats: Dict, iteration: int):
        """记录GCN相关指标"""
        if not self.logger:
            return
            
        try:
            self.log_scalar("GCN", "Alpha", gcn_alpha, iteration)
            
            for key, value in reward_propagation_stats.items():
                self.log_scalar("GCN", key, value, iteration)
                
        except Exception as e:
            print(f"Failed to log GCN metrics: {e}")
    
    def log_environment_info(self, env_name: str, num_processes: int, 
                           num_steps: int, num_frames: int):
        """记录环境信息"""
        if not self.task:
            return
            
        try:
            env_config = {
                "environment": env_name,
                "num_processes": num_processes,
                "num_steps": num_steps,
                "total_frames": num_frames
            }
            
            self.task.connect(env_config, name="Environment Config")
            print("Environment info logged to ClearML")
            
        except Exception as e:
            print(f"Failed to log environment info: {e}")
    
    def finish(self):
        """完成任务"""
        if self.task:
            try:
                self.task.close()
                print("ClearML task completed")
            except Exception as e:
                print(f"Failed to close ClearML task: {e}")

def create_clearml_tracker(args, task_name: str = None) -> ClearMLTracker:
    """
    创建ClearML跟踪器的便捷函数
    
    Args:
        args: 训练参数
        task_name: 任务名称，如果为None则自动生成
        
    Returns:
        ClearMLTracker实例
    """
    if task_name is None:
        # 自动生成任务名称
        env_name = getattr(args, 'env_name', 'unknown')
        algo = getattr(args, 'algo', 'unknown')
        seed = getattr(args, 'seed', 0)
        task_name = f"{env_name}_{algo}_seed{seed}"
    
    # 创建跟踪器
    tracker = ClearMLTracker(
        project_name="Phi-GCN",
        task_name=task_name
    )
    
    # 记录超参数
    tracker.log_hyperparameters(args)
    
    return tracker