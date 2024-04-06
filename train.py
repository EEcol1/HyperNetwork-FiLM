import random
import torch
assert torch.cuda.is_available(), "CUDA device not detected!"

import gymnasium as gym

from metaworld.envs import ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE

import numpy as np


import sac
from sac import ReplayBuffer,SACPolicy,Actor,ContinuousCritic
from sac import collect_rollouts,update_networks,evaluate_policy,evaluate_crl_metrics
from utils import CW5_sequence
from normalization import RunningMeanStd
import wandb

seed=0
sac.set_random_seed(seed)


wandb_log=True
nb_tasks=5
state_dim = 39
action_dim = 4
hidden_dim= 400
num_hidden_layers=2
hnet_hidden_dims=[100,100]
task_emb_dim=96
# chunk_emb_dim=96
# chunk_dim=42000
buffer_size = int(2e6)
batch_size = 1028
num_steps = int(1e6)
log_frequency = 2000
num_eval_episodes = 10
learning_rate = 3e-4
learning_rate_entropy=3e-4
tau=0.01
gamma=0.99
target_update_interval= 2
continual=True
config={
    "task_id": 0,
    "env_name" : f"{CW5_sequence[0]}-v2-goal-observable",
    "nb_tasks": nb_tasks,
    "state_dim" : state_dim,
    "action_dim" : action_dim,
    "hidden_dim" :hidden_dim ,
    "num_hidden_layers":num_hidden_layers,
    "hnet_hidden_dims":hnet_hidden_dims,
    "task_emb_dim": task_emb_dim,
    # "chunk_emb_dim":chunk_emb_dim,
    # "chunk_dim":chunk_dim,
    "buffer_size" : buffer_size,
    "batch_size" :batch_size,
    "num_steps" : num_steps,
    "log_frequency" : log_frequency,
    "num_eval_episodes" : num_eval_episodes,
    "learning_rate" : learning_rate,
    "learning_rate_entropy": learning_rate_entropy,
    "tau": tau,
    "gamma": gamma,
    "target_update_interval": target_update_interval,
    "seed": seed,
    "continual":continual
}

state_stats_normalizer=RunningMeanStd(num_tasks=nb_tasks,shape=(state_dim,))
reward_stats_normalizer=RunningMeanStd(num_tasks=nb_tasks,shape=(1,))

normalizers={
    "state_stats":state_stats_normalizer,
    "reward_stats":reward_stats_normalizer
}

for task_id in range(nb_tasks):

    env_name=CW5_sequence[task_id]
    print(task_id, env_name)
    
    config["task_id"]=task_id
    config["env_name"]=f"{env_name}-v2-goal-observable"

    #initialize environment
    env_name=config["env_name"]
    if wandb_log==True:
        wandb.init(project='film',config=config)
        wandb.run.name = f"task{task_id}-{CW5_sequence[task_id]}-te{task_emb_dim}-hnet_dim{hnet_hidden_dims}"

    # Initialize environment
    #env = gym.make(env_name) # Do not wrap with RecordVideo or it will cause problems later
    # ml1 = metaworld.ML1(env_name,seed=seed)
    # env = ml1.train_classes[env_name](render_mode=None)
    # task = random.choice(ml1.train_tasks)
    # env.set_task(task)

    env=ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[env_name](render_mode=None,seed=seed)
    env.model.cam_pos[2]=[0.75,0.075,0.7]
    env._freeze_rand_vec= False
    env= gym.wrappers.TimeLimit(env,max_episode_steps=200)

    # Initialize buffer
    buffer = ReplayBuffer(buffer_size, state_dim, action_dim)


    # Initialize policy and CUDA device (GPU or CPU)
    device = torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")
    policy = SACPolicy(state_dim, action_dim,nb_tasks,num_hidden_layers,task_emb_dim,hidden_dim,hnet_hidden_dims).to(device) 


    #actor_theta_optimizer = torch.optim.Adam(policy.actor.hnet.theta, lr=learning_rate) #hnet theta
    #actor_emb_optimizer = torch.optim.Adam([torch.nn.Parameter(policy.actor.hnet.task_emb.weight[task_id],requires_grad=True)],lr=learning_rate) 
    #actor_emb_optimizer = torch.optim.Adam([policy.actor.hnet.task_emb[task_id]],lr=learning_rate) 
    if task_id ==0:
        actor_optimizer=torch.optim.Adam(policy.actor.hnet.parameters(),lr=learning_rate)
    else:
        actor_optimizer=torch.optim.Adam(policy.actor.hnet.element_parameters,lr=learning_rate)    

    critic_optimizer = torch.optim.Adam(policy.critic.parameters(),lr=learning_rate) #
    entropy_coefficient_optimizer = torch.optim.Adam([policy.log_entropy_coefficient], lr=learning_rate_entropy)
    optimizers = {
        # "actor_theta": actor_theta_optimizer,
        # "actor_emb" : actor_emb_optimizer,
        "actor" :actor_optimizer,
        "critic": critic_optimizer,
        "entropy_coefficient": entropy_coefficient_optimizer
    }


    # # Initialize policy and CUDA device (GPU or CPU)
    # device = torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")
    # policy = SACPolicy(state_dim, action_dim,nb_tasks,num_hidden_layers,task_emb_dim,chunk_emb_dim,chunk_dim,hidden_dim).to(device) 
    #if task_id==0 else torch.load(f"{CW5_sequence[task_id-1]}-v2-goal-observable-policy.pt")
    if task_id>0 and continual==True:
        print("loading previous checkpoint...")
        ckpt=torch.load(f"ckpt/{CW5_sequence[task_id-1]}-v2-goal-observable-policy.pt")
        for key,value in policy.state_dict().items():
            assert(value.shape==ckpt[key].shape)
        pretrained_dict=ckpt
        new_model_dict=policy.state_dict()
        pretrained_dict={k:v for k,v in pretrained_dict.items() if k[0:5]=="actor"}
        for key in pretrained_dict:
            print(f"{key}-updated")
        new_model_dict.update(pretrained_dict)
        policy.load_state_dict(new_model_dict)
        state_stats=torch.load(f"norm/{CW5_sequence[task_id-1]}-v2-goal-observable-state.pt")
        reward_stats=torch.load(f"norm/{CW5_sequence[task_id-1]}-v2-goal-observable-reward.pt")
        state_stats_normalizer.load_state_dict(state_stats)
        reward_stats_normalizer.load_state_dict(reward_stats)

        #print("initializing optimizers for the new task...")
        #actor_emb_optimizer = torch.optim.Adam([policy.actor.hnet.task_emb[task_id]],lr=learning_rate) 
        #optimizers["actor_emb"]=actor_emb_optimizer
        
        #policy.critic = ContinuousCritic(state_dim, action_dim, num_hidden_layers, hidden_dim)
        #policy.critic_target = ContinuousCritic(state_dim, action_dim, num_hidden_layers, hidden_dim) 
        #policy.critic_target.load_state_dict(policy.critic.state_dict())
            
    # Define optimizer
    # actor_optimizer = torch.optim.Adam(policy.actor.hnet.parameters(), lr=learning_rate)
    # critic_optimizer = torch.optim.Adam(policy.critic.hnet_param, lr=learning_rate)
    # entropy_coefficient_optimizer = torch.optim.Adam([policy.log_entropy_coefficient], lr=learning_rate_entropy)
    # optimizers = {"actor": actor_optimizer, "critic": critic_optimizer, "entropy_coefficient": entropy_coefficient_optimizer}

    ##need to change input parameters
    #print(policy.actor.hnet.theta)
    #print(policy.actor.hnet.task_emb)

    # actor_theta_optimizer = torch.optim.Adam(policy.actor.hnet.theta, lr=learning_rate) #hnet theta
    # #actor_emb_optimizer = torch.optim.Adam([torch.nn.Parameter(policy.actor.hnet.task_emb.weight[task_id],requires_grad=True)],lr=learning_rate) 
    # actor_emb_optimizer = torch.optim.Adam([policy.actor.hnet.task_emb[task_id]],lr=learning_rate) 
    
   
    # critic_optimizer = torch.optim.Adam(policy.critic.parameters(),lr=learning_rate) #
    # entropy_coefficient_optimizer = torch.optim.Adam([policy.log_entropy_coefficient], lr=learning_rate_entropy)
    # optimizers = {
    #     "actor_theta": actor_theta_optimizer,
    #     "actor_emb" : actor_emb_optimizer,
    #     "critic": critic_optimizer,
    #     "entropy_coefficient": entropy_coefficient_optimizer
    # }
    

    task_id=task_id*torch.ones(1,dtype=torch.long,device=device)    
    
    # TRAINING LOOP
    #total_entropy_coefficient_loss, total_critic_loss, total_actor_loss = 0, 0, 0
    total_entropy_coefficient_loss, total_critic_loss, total_actor_loss = 0, 0, 0
    state, info = env.reset()
    for t in range(num_steps):
        # Rollout
        policy._set_to_eval()
        state,reward = collect_rollouts(buffer,state, env, policy, device,normalizers, task_id, random_step=t < 2000)
        
        normalizers["state_stats"].update(task_id,state)
        normalizers["reward_stats"].update(task_id,reward)
        
        
        if len(buffer) < 2000: # 샘플 개수가 batch_size 만큼 안 될시 네트워크 업데이트 스킵
            continue
        
        
        # Optimization steptask
        policy._set_to_train()
        entropy_coefficient_loss, critic_loss, actor_loss = update_networks(buffer, policy, device, optimizers,normalizers, task_id, batch_size=batch_size, gamma=gamma,tau=tau, update_target=(t%target_update_interval==0),continual=continual,log=(t%1000==0))
        #actor_loss=actor_loss_task+actor_loss_reg

        # Log results
        total_entropy_coefficient_loss += entropy_coefficient_loss / log_frequency
        total_critic_loss += critic_loss / log_frequency
        #total_actor_loss += actor_loss / log_frequency
        total_actor_loss += actor_loss/log_frequency
        #success_rate=info["success"]
        if (t + 1) % log_frequency == 0:
            policy._set_to_eval()
            total_score ,success_rate= evaluate_policy(policy, device,normalizers, env_name, task_id, num_episodes=num_eval_episodes)
            print(f"Step: {t + 1}", end="\t")
            print(f"Entropy Coefficient Loss: {total_entropy_coefficient_loss:.4f}", end="\t")
            print(f"Critic Loss: {total_critic_loss:.4f}", end="\t")
            #print(f"Actor Loss: {total_actor_loss:.4f}", end="\t")
            print(f"Actor Loss:{total_actor_loss:.4f}",end="\t")
            print(f"Total score: {total_score:.4f}",end="\t")
            print(f"Success Rate: {success_rate:.2f}")
            if wandb_log==True:
                wandb.log(
                    {
                    "Step": t+1,
                    "Entropy Coefficient Loss":total_entropy_coefficient_loss,
                    "Critic Loss": total_critic_loss,
                    "Actor Loss": total_actor_loss,
                    "Total score": total_score,
                    "success rate": success_rate
                    }
                )
            #total_entropy_coefficient_loss, total_critic_loss, total_actor_loss = 0, 0, 0
            total_entropy_coefficient_loss,total_critic_loss,total_actor_loss=0,0,0
        if (t+1) % 10000 ==0:
            policy._set_to_eval()
            success_rate_list,average_performance=evaluate_crl_metrics(policy,device,normalizers,nb_tasks,num_episodes=num_eval_episodes)
            print(f"average_performance: {average_performance:.2f}")
            if wandb_log==True:
                wandb.log(
                    {**{
                        f"success rate-{ii}": success_rate for ii, success_rate in enumerate(success_rate_list)
                    },
                    **{
                        f"average_performance": average_performance
                    }
                    }
                )
        if (t+1) % int(1e5) ==0:
            torch.save(policy.state_dict(),f"ckpt/{env_name}-policy-temp-{(t+1)//int(1e5)}.pt")
    # Close environment
    env.close()
    #model save
    torch.save(policy.state_dict(),f"ckpt/{env_name}-policy.pt")
    torch.save(state_stats_normalizer.state_dict(),f"norm/{env_name}-state.pt")
    torch.save(reward_stats_normalizer.state_dict(),f"norm/{env_name}-reward.pt")
    if wandb_log==True:
        wandb.finish()
