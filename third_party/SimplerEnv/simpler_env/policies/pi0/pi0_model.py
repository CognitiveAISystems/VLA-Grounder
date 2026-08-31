import sys
from pathlib import Path
import torch
import draccus

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent.resolve())) 

from agent.configuration_pipeline import TrainPipelineConfig
from experiments.policies.policy_wrapper import LeRobotPolicyWrapper
from lerobot.common.policies.pi0.modeling_pi0 import PI0Policy as LeRobotPI0Policy

class PI0Policy:
    def __init__(self, all_args, device_id: int):
        self.args = all_args
        self.device_id = device_id
        
        PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
        config_path = str(PROJECT_ROOT / "SimplerEnv" / "pi0_finetune_bridge_ev.yaml")

        def load_config_only():
            test_args = [
                "--config_path", config_path,
                "--seed", str(all_args.seed), 
                "--use_bf16", "True",
            "--use_wandb", "False",
            ]
    
            cfg = draccus.parse(TrainPipelineConfig, args=test_args)
            return cfg
        
        pipeline_cfg = load_config_only()
        
        self.wrapper = LeRobotPolicyWrapper(pipeline_cfg, LeRobotPI0Policy)
        self.wrapper._initialze_model_server(all_args.vla_path)
        self.wrapper.env_adapter = self.wrapper._initialize_env_adapter()
        
    def get_action(self, x: dict, deterministic=True):
        if isinstance(x["image"], torch.Tensor):
            images = x["image"].cpu().numpy() # [B, H, W, 3]
        else:
            images = x["image"]
        tasks = x["task_description"]
        obs = x.get("pi_0")
        
        batch_size = images.shape[0]
        
        element = {
            "observation.images.top": images,
            "observation.state": obs['eef_pos'],
            "task": tasks,
        }
        actions = self.wrapper.select_action(element)
        
        # Ensure actions is [B, D] if T=1
        if actions.ndim == 3 and actions.shape[1] == 1:
            actions = actions.squeeze(1)
            
        values = torch.zeros(batch_size, 1).to(self.wrapper.device)
        logprobs = torch.zeros(batch_size, 1).to(self.wrapper.device)
        
        actions_tensor = torch.tensor(actions).to(self.wrapper.device)
        
        return values, actions_tensor, logprobs

    def prep_rollout(self):
        self.wrapper.model.eval()
        
    def prep_training(self):
        self.wrapper.model.train()
        
    def save(self, path):
        pass
