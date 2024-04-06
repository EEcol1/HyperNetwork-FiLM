import torch
import sac
import random
import gymnasium as gym

from metaworld.envs import ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE
import numpy as np

def set_random_seed(random_seed: int):
    seed=random_seed
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)



@torch.no_grad()
def evaluate_policy(policy_gt,policy, device, env_name,task_id, num_episodes=10, verbose=False):
    # Make environment
    #env = gym.make("Pendulum-v1")
    env=ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[env_name](render_mode=None,seed=0)
    env.model.cam_pos[2]=[0.75,0.075,0.7]
    env._freeze_rand_vec= False
    env= gym.wrappers.TimeLimit(env,max_episode_steps=200)


    policy_gt._set_to_eval()
    policy._set_to_eval()
    
    #change task_id as tensor
    #task_id=torch.tensor([task_id],dtype=torch.int64).to(device)


    # Reset environment
    state, info = env.reset()

    total_score = 0
    total_success=0
    mse_sum=0
        
    for e in range(num_episodes):
        total_reward = 0
        mse=0
        while True:
            # Get action
            state_tensor = torch.from_numpy(state.astype(np.float32)).unsqueeze(0).to(device)
            #(4,)
            action_tensor_gt = policy_gt.actor(state_tensor, task_id, deterministic=True)
            #(4,)
            action_tensor=policy.actor(state_tensor,task_id,deterministic=True)
            #action = 2 * action_tensor.detach().cpu().numpy().reshape(-1)
            # if temp<=10:
            #     print(f"state: {state_tensor[0][0:4]}")
            #     print(f"gt: {action_tensor_gt}")
            #     print(f"now: {action_tensor}")
                # print((action_tensor_gt-action_tensor))
            # temp+=1    

            
            mse+=torch.sum((action_tensor_gt-action_tensor)**2)
            action = action_tensor_gt.detach().cpu().numpy().reshape(-1)
            
            # Environment step
            state, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

            # Episode ends
            if terminated or truncated:
                if verbose:
                    print(f"[Episode {e + 1}] Score: {total_reward:.4f} Success: {info['success']}")
                    
                total_success+=info["success"]
                break
        
        mse/=800
        #print(f"[episode{e}] MSE: {mse}")
        mse_sum+=mse
        # Reset environment
        total_score += total_reward
        state, info = env.reset()
    
    # Close environment
    env.close()


    print(mse_sum/num_episodes)
#    return total_score / num_episodes, total_success / num_episodes




CW10_sequence=[
    "hammer", 
    "push-wall", 
    "faucet-close",
    "handle-press-side", 
    "window-close"
]
device = torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")
def eval(task_id,task_id_test,num_episodes):    
    env_name=f"{CW10_sequence[task_id]}-v2-goal-observable"
    ckpt=torch.load(f"ckpt/{CW10_sequence[task_id]}-v2-goal-observable-policy.pt")
    policy_gt=sac.SACPolicy(39,4,5,2,192,400,[100,100],2).to(device)
    policy_gt.load_state_dict(ckpt)

    policy_test_list=[]
    for i in range(10):
        policy_test=(sac.SACPolicy(39,4,5,2,192,400,[100,100],2).to(device))
        policy_test.load_state_dict(torch.load(f"ckpt/{CW10_sequence[task_id_test]}-v2-goal-observable-policy-temp-{i+1}.pt"))
        policy_test_list.append(policy_test)



    #print(evaluate_policy(policy,device,env_name,task_id,verbose=True))
    #print(evaluate_policy(policy1,device,env_name,task_id,verbose=True))
    #print(evaluate_policy(policy2,device,env_name,task_id,verbose=True))



    env=ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[env_name](render_mode=None,seed=0)
    env.model.cam_pos[2]=[0.75,0.075,0.7]
    env._freeze_rand_vec= False
    env= gym.wrappers.TimeLimit(env,max_episode_steps=200)


    policy_gt._set_to_eval()
    for p in policy_test_list:
        p._set_to_eval()
    #change task_id as tensor
    #task_id=torch.tensor([task_id],dtype=torch.int64).to(device)
    test_list=[]
    for i,p in enumerate(policy_test_list):
        print(f"temp-{i+1} ckpt test")
        evaluate_policy(policy_gt,p,device,env_name,task_id,num_episodes)
        
    # # Reset environment
    # state, info = env.reset()
    # state_tensor = torch.from_numpy(state.astype(np.float32)).unsqueeze(0).to(device)
    # #check if task_emb of previous task remains same
    # print("check task_emb")
    # print(policy.actor.hnet.task_emb[0])
    # print(policy1.actor.hnet.task_emb[0])
    # print(policy2.actor.hnet.task_emb[0])

    # #check if it genereates similar theta
    # print("check if making similar theta")
    # print(policy.actor.hnet(0)[0][0][0:10])
    # print(policy1.actor.hnet(0)[0][0][0:10])
    # print(policy2.actor.hnet(0)[0][0][0:10])

    # #check if it generates same output
    # print("check if it generates same output")
    # print(policy.actor.mnet(state_tensor,policy.actor.hnet(0)))
    # print(policy1.actor.mnet(state_tensor,policy1.actor.hnet(0)))
    # print(policy2.actor.mnet(state_tensor,policy2.actor.hnet(0)))
    
# for i in range(5):
#     for j in range(i+1,5):
#         print(f"ground truth {i}, {j}번 째 task")
#         eval(i,j)
set_random_seed(0)
eval(1,2,10)
