from pathlib import Path

import torch.nn as nn
import numpy as np
import sys
import os
sys.path.append("/home/keyi/Documents/VsCodeP/research/retarget/bimanual")
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..','..','assistive_teleop'))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..','assistive_stable_baselines3'))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__))))


from assistive_teleop.hand_env_utils.arg_utils import *
from assistive_teleop.hand_env_utils.wandb_callback import WandbCallback, setup_wandb
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv
from stable_baselines3.ppo import PPO
from assistive_teleop.envs.rl_env.assistive_teleop_bimanual_env import AssistiveTeleopBimanualEnv
from assistive_teleop.envs.real_world import task_setting

from datetime import datetime

def create_env(use_visual_obs, use_gui=False, is_eval=False, obj_scale=1.0, obj_name=None,
               reward_args=np.zeros(3), data_id=0, randomness_scale=1, pc_noise=True):
    import os
    from assistive_teleop.envs.rl_env.assistive_teleop_bimanual_env import AssistiveTeleopBimanualEnv
    # from hand_teleop.real_world import task_setting
    from assistive_teleop.envs.sim_env.constructor import add_default_scene_light
    frame_skip = 5
    # env_params = dict(reward_args=reward_args, object_scale=obj_scale, object_name=obj_name, data_id=data_id,
    #                   use_gui=use_gui, frame_skip=frame_skip, no_rgb=True)
    # if is_eval:
    #     env_params["no_rgb"] = False 
    #     env_params["need_offscreen_render"] = True

    # # Specify rendering device if the computing device is given
    # if "CUDA_VISIBLE_DEVICES" in os.environ:
    #     env_params["device"] = "cuda"
    
    config_path = "/home/keyi/Documents/VsCodeP/research/retarget/bimanual/assistive_teleop/configs/default.yaml"
    data_path = "/home/keyi/Documents/VsCodeP/research/retarget/bimanual/assistive_teleop/rsc"
    task = 'box_grab'  
    env_params = dict(config_path = config_path, data_path = data_path, task = task, reward_args = reward_args, data_id=data_id)
    if is_eval:
        env_params["no_rgb"] = False 
        env_params["need_offscreen_render"] = True

    # Specify rendering device if the computing device is given
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        env_params["device"] = "cuda"
    env = AssistiveTeleopBimanualEnv(**env_params)

    if use_visual_obs:
        raise NotImplementedError
    
    if is_eval:
        env.setup_camera_from_config(task_setting.CAMERA_CONFIG["viz_only"])
        add_default_scene_light(env.scene, env.renderer)
    
    return env


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=100)
    parser.add_argument('--workers', type=int, default=10)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--ep', type=int, default=10)
    parser.add_argument('--bs', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=100)
    parser.add_argument('--iter', type=int, default=2000)
    parser.add_argument('--randomness', type=float, default=1.0)
    parser.add_argument('--exp', type=str, default="box")
    parser.add_argument('--reward', type=float, nargs="+", default=[1, 0.05, 0.01])
    parser.add_argument('--objscale', type=float, default=1.0)
    parser.add_argument('--objcat', type=str, default="auto")
    parser.add_argument('--objname', type=str, default="auto")
    parser.add_argument('--dataid', type=int, default=0)

    args = parser.parse_args()
    randomness = args.randomness
    data_id = args.dataid
    now = datetime.now()
    exp_keywords = [args.exp, str(args.dataid)]
    if str(args.objcat) != "auto":
        exp_keywords.append(str(args.objcat))
    if str(args.objname) != "auto":
        exp_keywords.append(str(args.objname))
    exp_keywords.append(",".join(str(i) for i in args.reward))
    horizon = 200
    env_iter = args.iter * horizon * args.n
    reward_args = args.reward
    data_id = args.dataid
    assert(len(reward_args) >= 3)
    obj_scale = args.objscale
    obj_name = (args.objcat, args.objname)

    config = {
        'n_env_horizon': args.n,
        'update_iteration': args.iter,
        'total_step': env_iter,
        'randomness': randomness,
    }

    exp_name = "-".join(exp_keywords)
    result_path = Path("./results") / exp_name
    result_path.mkdir(exist_ok=True, parents=True)
    wandb_run = setup_wandb(config, "-".join([exp_name, now.strftime("(%Y/%m/%d,%H:%M)")]), tags=["state", "imitation"])

    def create_env_fn():
        environment = create_env(use_visual_obs=False, obj_scale=obj_scale, obj_name=obj_name,
                                 reward_args=reward_args, data_id=data_id, randomness_scale=randomness)
        return environment
    
    def create_eval_env_fn():
        environment = create_env(use_visual_obs=False, obj_scale=obj_scale, obj_name=obj_name,
                                 reward_args=reward_args, data_id=data_id, is_eval=True, randomness_scale=randomness)
        return environment

    env = SubprocVecEnv([create_env_fn] * args.workers, "spawn")
    # env = AssistiveTeleopBimanualEnv(
    #     "/home/keyi/Documents/VsCodeP/research/retarget/bimanual/assistive_teleop/configs/default.yaml"
    # )
    
    print(env.observation_space, env.action_space)

    model = PPO("MlpPolicy", env, verbose=1,
                n_epochs=args.ep,
                n_steps=(args.n // args.workers) * horizon,
                learning_rate=args.lr,
                batch_size=args.bs,
                seed=args.seed,
                policy_kwargs={'activation_fn': nn.ReLU},
                min_lr=args.lr,
                max_lr=args.lr,
                adaptive_kl=0.02,
                target_kl=0.2,
                )

    model.learn(
        total_timesteps=int(env_iter),
        callback=WandbCallback(
            model_save_freq=50,
            model_save_path=str(result_path / "model"),
            eval_env_fn=create_eval_env_fn,
            eval_freq=100,
            eval_cam_names=["relocate_viz"],
        ),
    )
    wandb_run.finish()