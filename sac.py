import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
assert torch.cuda.is_available(), "CUDA device not detected!"

import gymnasium as gym

import metaworld
from metaworld.envs import ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE

from typing import NamedTuple, Union
import numpy as np
import model
import utils
from utils import CW5_sequence

seed=0
def set_random_seed(random_seed: int):
    seed=random_seed
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    model.set_random_seed(random_seed)

class ReplayBufferSamples(NamedTuple):
    states: Union[np.array, torch.Tensor]
    actions: Union[np.array, torch.Tensor]
    next_states: Union[np.array, torch.Tensor]
    dones: Union[np.array, torch.Tensor]
    rewards: Union[np.array, torch.Tensor]

# Buffer에 데이터가 하나씩만 들어오는 경우만 고려
# Initialize 할 때 numpy array, dtype=float32, 로 만드는것 추천
# 데이터 추가할 때 마다, current_index += 1 을 해서, 현재 샘플 개수 카운트, 최대 사이즈 도달 시 full=True
# ReplayBufferSamples 클래스 사용하면 유용, sample_batch output도 ReplayBufferSamples 형태
# batch 샘플할 때 torch.Tensor 로 미리 변환하고 device에 올린 상태로 쓸 것
# __len__(self) 함수는 len(buffer) 의 output을 return 하게 하는 것 (현재 저장한 샘플 수)

class ReplayBuffer():
    def __init__(self, buffer_size, state_dim=39, action_dim=4):
        self.buffer_size = buffer_size
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.reset()

    def reset(self):
        self._states = np.zeros((self.buffer_size,self.state_dim),dtype=np.float32)
        self._actions = np.zeros((self.buffer_size,self.action_dim),dtype=np.float32)
        self._next_states = np.zeros((self.buffer_size,self.state_dim),dtype=np.float32)
        self._rewards = np.zeros((self.buffer_size,1),dtype=np.float32)
        self._dones = np.zeros((self.buffer_size,1),dtype=np.float32)
        self._current_index = 0
        self._full = False

    def add(self, data):
        assert isinstance(data, ReplayBufferSamples)
        if self._full is True:
            self._current_index=0
        self._states[self._current_index]=data.states
        self._actions[self._current_index]=data.actions
        self._next_states[self._current_index]=data.next_states
        self._rewards[self._current_index]=data.rewards
        self._dones[self._current_index]=data.dones
        self._current_index+=1
        if self._current_index == self.buffer_size:
            self._full=True

    def sample_batch(self, batch_size, device=None,task_id=0,state_stats=None,reward_stats=None):
        if device is None:
            device=torch.device('cuda:0' if torch.cuda.is_available else 'cpu')
        random_idx=np.random.randint(0,self._current_index,size=batch_size)
        replay_data = [
            self._states[random_idx] if state_stats is None else state_stats.normalize_batch(task_id, self._states[random_idx]),
            self._actions[random_idx],
            self._next_states[random_idx] if state_stats is None else state_stats.normalize_batch(task_id, self._next_states[random_idx]),
            self._dones[random_idx],
            self._rewards[random_idx] if reward_stats is None else reward_stats.normalize_batch(task_id, self._rewards[random_idx])
        ]
        def to_tensor(np_data):
            return torch.from_numpy(np_data.astype(np.float32)).to(device)
        replay_data_tensor=list(map(to_tensor,replay_data))

        return ReplayBufferSamples(*replay_data_tensor)


    def __len__(self):
        return self._current_index
    




# SAC 에서는 Gaussian -> SquashedGaussian을 이용하여 action을 tanh() 을 통해서 바운드한다. [-∞,∞] -> [-1,1]
# 여기에 나와있는 Gaussian distribution 을 만들기위하여 torch.distributions.Normal을 주로 사용
# 여러 가지 implementation이 있지만, Gaussian의 mean과 log_std 를 output하는 네트워크를 정의한 뒤에 Gaussian distribution을 만들것
# **Policy 부분은 수정할 필요 없으나, 자세히 읽는 것 추천

class Actor(nn.Module):
    
    def __init__(self,state_dim,action_dim,nb_tasks,num_hidden_layers,task_emb_dim, hidden_dim=400,hnet_hidden_dims=[10,10]):
        super().__init__()
        self.nb_tasks= nb_tasks
        self.action_dim=action_dim
        self.mnet=model.MainNetwork(state_dim,2*action_dim,hidden_dims=[hidden_dim]*num_hidden_layers)
        target_shapes = self.mnet._param_shapes
        self.hnet=model.HyperNetwork(
            target_shapes=target_shapes, 
            num_tasks=nb_tasks, 
            task_emb_dim=task_emb_dim,
            hidden_dims=hnet_hidden_dims
        )

    def forward(self, states,task_id, deterministic=False):
        # Construct Gaussian distribution: π(a|s) = N(μ,σ^2⋅I)
       
        mnet_params=self.hnet(task_id=task_id)

        x=self.mnet(states,mnet_params)

        mean_actions,log_std_actions=x[:,:self.action_dim:], x[:,self.action_dim:]

        log_std_actions=torch.clamp(log_std_actions,-20,2)
        std_actions=torch.exp(log_std_actions)
        dist_actions=Normal(mean_actions,std_actions)


        # Get actions: a = argmax π(a|s) or a ~ π(a|s)
        if deterministic:
            actions = torch.tanh(mean_actions) # Squashed Gaussian
            return actions
        else:
            gaussian_actions = dist_actions.rsample() # torch.distributions을 보면 보통 sample()과 rsample()이 있는데 rsample() 만 backpropagation 가능 (reparameterization trick 참고)
            actions = torch.tanh(gaussian_actions)

        # Compute log-probabilities: log π(a|s)
        gaussian_log_probs = dist_actions.log_prob(gaussian_actions).sum(dim=1, keepdim=True)
        log_probs = gaussian_log_probs - torch.sum(torch.log(1 - actions ** 2 + 1e-6), dim=1, keepdim=True)
        #print(actions.shape) #!!
        return actions, log_probs

class ContinuousCritic(nn.Module):
    
    def __init__(self, state_dim,action_dim,num_hidden_layers,hidden_dim=400,num_critics=2):
        super().__init__()
        self.num_critics=num_critics
        self.q_networks = []
        hidden_dims=[state_dim+action_dim] + [hidden_dim]*num_hidden_layers

        for idx in range(num_critics):
            q_net=[]
            for layer in range(len(hidden_dims)-1):
                q_net.append(nn.Linear(hidden_dims[layer],hidden_dims[layer+1]))
                q_net.append(nn.ReLU())
            q_net.append(nn.Linear(hidden_dims[-1],1))
            nn.init.normal_(q_net[-1].weight,mean=0.,std=1.)
            nn.init.normal_(q_net[-1].bias,mean=0.,std=1.)
            q_net= nn.Sequential(*q_net)
            self.add_module(f"qf{idx}", q_net)
            self.q_networks += [q_net]
        
    

    def forward(self, states, actions):
        inputs = torch.cat([states, actions], dim=1)
        return tuple(q_net(inputs) for q_net in self.q_networks)
        #return tuple(q_net(inputs,task_id) for q_net in self.q_networks)

class SACPolicy(nn.Module):
    def __init__(self, state_dim, action_dim, nb_tasks, num_hidden_layers=2, task_emb_dim=96, hidden_dim=400, hnet_hidden_dims=[10,10],num_critics=2):
        super().__init__()
        # Networks
        self.actor = Actor(state_dim, action_dim, nb_tasks, num_hidden_layers,task_emb_dim,hidden_dim,hnet_hidden_dims)
        self.critic = ContinuousCritic(state_dim, action_dim, num_hidden_layers, hidden_dim, num_critics)
        self.critic_target = ContinuousCritic(state_dim, action_dim, num_hidden_layers, hidden_dim, num_critics) 
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        # Entropy coefficient
        self.log_entropy_coefficient = nn.Parameter(torch.log(0.1 * torch.ones(1)), requires_grad=True)
        self.target_entropy = -4.
        

    def _set_to_train(self):
        self.actor.train()
        self.critic.train()
        self.critic_target.eval()

    def _set_to_eval(self):
        self.actor.eval()
        self.critic.eval()
        self.critic_target.eval()

    @torch.no_grad()
    def update_target_networks(self, tau=0.01):
        for params, target_params in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_params.data.mul_(1 - tau)
            torch.add(target_params.data, params.data, alpha=tau, out=target_params.data)

@torch.no_grad()
def collect_rollouts(buffer, state, env, policy, device, normalizers, task_id, random_step=False):
    # Sample action
    # 초반에는 policy를 따라 action 샘플하는 것보다 랜덤으로 샘플할것
    # 중요: 실제로 env에 취할 action바운드는 [-2, 2] 이지만 policy output & buffer에 들어갈 action의 바운드는 [-1, 1] (SquashedGaussian 참고)
    
    #change task_id as tensor
    #task_id=torch.tensor([task_id],dtype=torch.int64).to(device)

    if random_step:
        action=env.action_space.sample()
        
    else:
        norm_state=normalizers["state_stats"].normalize(task_id,state)
        action,_=policy.actor(torch.tensor(norm_state,dtype=torch.float32).unsqueeze(0).to(device),task_id) #input: (3,) output:(1,)
        action=action.squeeze(0)
        action=action.cpu().numpy()
        


    # Environment step
    next_state ,reward, terminated, truncated, _=env.step(action)
    done=(terminated)

    # Add sample to buffer
    reward=np.array([reward])
    done=np.array([done])
    
    buffer.add(ReplayBufferSamples(state,action,next_state,done,reward))
    # Reset if done
    #print("state",state.shape, "action",action.shape,"next_state",next_state.shape,"reward",reward.shape,"done",done.shape) !!
    if truncated or terminated:
        next_state,_=env.reset()
        normalizers["state_stats"].normalize(task_id,state)

    return next_state,reward

def update_networks(buffer, policy, device, optimizers,normalizers, task_id, batch_size=512, gamma=0.99,tau=0.01, update_target=True,continual=False,log=False):
    # Sample batch
    batch=buffer.sample_batch(batch_size,device,task_id,normalizers["state_stats"],normalizers["reward_stats"])
    states,actions,next_states,rewards,dones=batch.states,batch.actions,batch.next_states,batch.rewards,batch.dones

    # Sample actions and compute their log-probabilities
    curr_actions,log_probs=policy.actor(states,task_id)

    # Compute entropy coefficient loss: J(α) = E_π[-α⋅(log π(a|s) + H_0)]
    entropy_coefficient = torch.exp(policy.log_entropy_coefficient.detach())
    entropy_coefficient_loss = -(policy.log_entropy_coefficient * (log_probs + policy.target_entropy).detach()).mean()

    # Update entropy coefficient
    optimizers["entropy_coefficient"].zero_grad()
    entropy_coefficient_loss.backward()
    optimizers["entropy_coefficient"].step()

    # calc_reg=task_id>0 and continual==True





    # Compute targets for critics: y = r + γ⋅(1-d)⋅(min Q(s',a') - α⋅log π(a'|s'))
    with torch.no_grad():
        next_actions,next_log_probs=policy.actor(next_states,task_id)
        next_q_values=policy.critic_target(next_states,next_actions)
        next_q_values=torch.cat(next_q_values,dim=1)
        next_q_value,_=torch.min(next_q_values,dim=1,keepdim=True)
        y=rewards+gamma*(1-dones)*(next_q_value-entropy_coefficient*next_log_probs) #compute T

    # Compute critic loss: J_Q(φ) = E_D[0.5⋅(Q(s,a) - y)^2]
    q_values=torch.cat(policy.critic(states,actions),dim=1) # compute Y
    critic_loss=(F.mse_loss(q_values[:,0].reshape(-1,1),y)+F.mse_loss(q_values[:,1].reshape(-1,1),y))*0.5 # 

    # Update critics
    optimizers["critic"].zero_grad()
    critic_loss.backward()
    torch.nn.utils.clip_grad_norm_(policy.critic.parameters(),1)
    optimizers["critic"].step()



    # Compute actor loss: J_π(θ) = E_D[E_π[α⋅log π(a|s) - Q(s,a)]]
    q_values_pi=torch.cat(policy.critic(states,curr_actions),dim=1)
    min_qf_pi,_=torch.min(q_values_pi,dim=1,keepdim=True)
    actor_loss=(entropy_coefficient*log_probs-min_qf_pi).mean()


    optimizers["actor"].zero_grad()
    actor_loss.backward()
    optimizers["actor"].step()

    # Update target networks
    if update_target:
        policy.update_target_networks(tau=tau)

    return entropy_coefficient_loss.item(), critic_loss.item(), actor_loss.item()







@torch.no_grad()
def evaluate_policy(policy, device, normalizers, env_name,task_id, num_episodes=10, verbose=False):
    # Make environment
    #env = gym.make("Pendulum-v1")
    env=ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[env_name](render_mode=None,seed=seed)
    env.model.cam_pos[2]=[0.75,0.075,0.7]
    env._freeze_rand_vec= False
    env= gym.wrappers.TimeLimit(env,max_episode_steps=200)


    policy._set_to_eval()
    

    # Reset environment
    state, info = env.reset()

    total_score = 0
    total_success=0
    for e in range(num_episodes):
        total_reward = 0
        while True:
            # Get action
            norm_state=normalizers["state_stats"].normalize(task_id,state)
            state_tensor = torch.from_numpy(norm_state.astype(np.float32)).unsqueeze(0).to(device)
            action_tensor = policy.actor(state_tensor, task_id, deterministic=True)
            #action = 2 * action_tensor.detach().cpu().numpy().reshape(-1)
            action = action_tensor.detach().cpu().numpy().reshape(-1)

            # Environment step
            state, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

            # Episode ends
            if terminated or truncated:
                if verbose:
                    print(f"[Episode {e + 1}] Score: {total_reward:.4f}")
                total_success+=info["success"]
                break
            
        # Reset environment
        total_score += total_reward
        state, info = env.reset()

    # Close environment
    env.close()

    return total_score / num_episodes, total_success / num_episodes






@torch.no_grad()
def evaluate_crl_metrics(policy,device,normalizers,num_tasks=10,num_episodes=10):
        # Make environment
    #env = gym.make("Pendulum-v1")
    total_success_list=[]
    policy._set_to_eval()
    for curr_task_id in range(num_tasks):
        
        env=ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[f"{CW5_sequence[curr_task_id]}-v2-goal-observable"](render_mode=None,seed=seed)
        env.model.cam_pos[2]=[0.75,0.075,0.7]
        env._freeze_rand_vec= False
        env= gym.wrappers.TimeLimit(env,max_episode_steps=200)

    # Reset environment
        state, info = env.reset()

        total_success=0
        for e in range(num_episodes):
            while True:
                # Get action
                norm_state=normalizers["state_stats"].normalize(curr_task_id,state)
                state_tensor = torch.from_numpy(norm_state.astype(np.float32)).unsqueeze(0).to(device)
                action_tensor = policy.actor(state_tensor, torch.tensor([curr_task_id],dtype=torch.int64).to(device), deterministic=True)
                #action = 2 * action_tensor.detach().cpu().numpy().reshape(-1)
                action = action_tensor.detach().cpu().numpy().reshape(-1)

                # Environment step
                state, reward, terminated, truncated, info = env.step(action)
                
                # Episode ends
                if terminated or truncated:
                    total_success+=info["success"]
                    break
                
            # Reset environment
            state, info = env.reset()

        total_success_list.append(total_success/num_episodes)
        # Close environment
        env.close()

    return total_success_list,  sum(total_success_list)/ num_tasks

