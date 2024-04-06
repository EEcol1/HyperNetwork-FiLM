import sac
from sac import SACPolicy,Actor,ContinuousCritic
import torch
import numpy as np
import gymnasium as gym
from metaworld.envs import ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE

seed=0
device = torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")
#sac.set_random_seed(seed)

check_point_id=2
task_id=1
CW10_sequence=[
    "hammer", 
    "push-wall", 
    "faucet-close", 
    "push-back", 
    "stick-pull", 
    "handle-press-side", 
    "push", 
    "shelf-place", 
    "window-close", 
    "peg-unplug-side"
]



env_name=f"{CW10_sequence[task_id]}-v2-goal-observable"

env=ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[env_name](render_mode="rgb_array",seed=seed)
env.model.cam_pos[2]=[0.75,0.075,0.7]
env._freeze_rand_vec= False
env= gym.wrappers.TimeLimit(env,max_episode_steps=200)
env=gym.wrappers.RecordVideo(env,f"{CW10_sequence[task_id]}-checkpoint-{CW10_sequence[check_point_id]}")
#record video

policy=torch.load(f"{CW10_sequence[check_point_id]}-v2-goal-observable-policy.pt")
task_id=torch.tensor([task_id],dtype=torch.int64).to(device)
policy._set_to_eval() # 생각보다 중요한 디테일. https://rabo0313.tistory.com/entry/Pytorch-modeltrain-modeleval-%EC%9D%98%EB%AF%B8 (이외에도 고려할 것이 많다.)

# Reset environment
state, info = env.reset()

total_reward = 0
device="cuda"
for t in range(1000):
    # Get action
    state_tensor = torch.from_numpy(state.astype(np.float32)).unsqueeze(0).to(device)
    with torch.no_grad():
        action_tensor = policy.actor(state_tensor,task_id, deterministic=True)
    #action = 2 * action_tensor.detach().cpu().numpy().reshape(-1)
    action = action_tensor.detach().cpu().numpy().reshape(-1)
    # Environment step
    state, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    

    # Render environment via virtual display
    env.render()

    if terminated:
        print("Terminated. {} steps".format(t + 1))
        break

    if truncated:
        print("Truncated. {} steps".format(t + 1))
        break

# Close environment
env.close()

print(info['success'])
print('Total Reward: {:.2f}'.format(total_reward))