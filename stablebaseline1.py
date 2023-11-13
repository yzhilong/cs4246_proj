import gym
import gym_dssat_pdi
import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from stable_baselines3 import PPO, A2C, DDPG, DQN, TD3, SAC, HER
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
import argparse

parser = argparse.ArgumentParser(
    prog='train selected crops',
)
parser.add_argument('-c', '--cultivar', default='maize', type=str)
parser.add_argument('-s', '--seed', default=123, type=int)
parser.add_argument('-r', '--random_weather', default=True, type=bool)
parser.add_argument('-m', '--mode', default='fertilization', type=str)
args = parser.parse_args()

print(args)


TOTAL_TIMESTEPS = 400_000

# helpers for action normalization
def normalize_action(action_space_limits, action):
    """Normalize the action from [low, high] to [-1, 1]"""
    low, high = action_space_limits
    return 2.0 * ((action - low) / (high - low)) - 1.0

def denormalize_action(action_space_limits, action):
    """Denormalize the action from [-1, 1] to [low, high]"""
    low, high = action_space_limits
    return low + (0.5 * (action + 1.0) * (high - low))

# Wrapper for easy and uniform interfacing with SB3
class GymDssatWrapper(gym.Wrapper):
    def __init__(self, env):
        super(GymDssatWrapper, self).__init__(env)

        self.action_low, self.action_high = self._get_action_space_bounds()

        # using a normalized action space
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(2,), dtype="float32")

        # using a vector representation of observations to allow
        # easily using SB3 MlpPolicy
        self.observation_space = gym.spaces.Box(low=0.0,
                                                high=np.inf,
                                                shape=env.observation_dict_to_array(
                                                    env.observation).shape,
                                                dtype="float32"
                                                )

        # to avoid annoying problem with Monitor when episodes end and things are None
        self.last_info = {}
        self.last_obs = None

    def _get_action_space_bounds(self):
        box = self.env.action_space['anfer']
        return box.low, box.high

    def _format_action(self, action):
        return { 'anfer': action[0] }

    def _format_observation(self, observation):
        return self.env.observation_dict_to_array(observation)

    def reset(self):
        return self._format_observation(self.env.reset())


    def step(self, action):
        # Rescale action from [-1, 1] to original action space interval
        denormalized_action = denormalize_action((self.action_low, self.action_high), action)
        formatted_action = self._format_action(denormalized_action)
        formatted_action['amir'] = 0
        obs, reward, done, info = self.env.step(formatted_action)

        # handle `None`s in obs, reward, and info on done step
        if done:
            obs, reward, info = self.last_obs, 0, self.last_info
        else:
            self.last_obs = obs
            self.last_info = info

        formatted_observation = self._format_observation(obs)
        print('reward', reward)
        return formatted_observation, reward, done, info

    def close(self):
        return self.env.close()

    def seed(self, seed):
        self.env.set_seed(seed)

    def __del__(self):
        self.close()

# Create environment
env_args = {
    'mode': args.mode,
    'seed': args.seed,
    'random_weather': args.random_weather,
    'cultivar': args.cultivar
}

env = GymDssatWrapper(gym.make('GymDssatPdi-v0', **env_args))
# Training arguments for PPO agent
ppo_args = {
    'gamma': 0.99,
    'learning_rate': 0.0003,
    'seed': 123,
}

# Create the agent
ppo_agent = PPO('MlpPolicy', env, **ppo_args)
a2c_agent = A2C('MlpPolicy', env, **ppo_args)
ddpg_agent = DDPG('MlpPolicy', env, **ppo_args)
# dqn_agent = DQN('MlpPolicy', env, **ppo_args)
td3_agent = TD3('MlpPolicy', env, **ppo_args)
sac_agent = SAC('MlpPolicy', env, **ppo_args)
# her_agent = HER('MlpPolicy', env, **ppo_args)

# Train for 400k timesteps
agent_names = 'PPO,A2C,DDPG,TD3,SAC'.split(',')
agents = [
    ppo_agent, a2c_agent, ddpg_agent, td3_agent, sac_agent
]
agent_name_mapping = {agent_name: agent for agent_name, agent in zip(agent_names, agents)}
for agent_name, agent in zip(agent_names, agents):
    print(f'Training {agent_name} agent...')
    agent.learn(total_timesteps=TOTAL_TIMESTEPS, progress_bar=True)
    print(f'Training done for {agent_name}')
    env.reset()

# Baseline agents for comparison
class NullAgent:
    """
    Agent always choosing to do no fertilization
    """
    def __init__(self, env):
        self.env = env

    def predict(self, obs, state=None, episode_start=None, deterministic=None):
        action = normalize_action((self.env.action_low, self.env.action_high), [0])
        return np.array([action], dtype=np.float32), obs


class ExpertAgent:
    """
    Simple agent using policy of choosing fertilization amount based on days after planting
    """
    fertilization_dic = {
        40: 27,
        45: 35,
        80: 54,
    }

    def __init__(self, env, normalize_action=False, fertilization_dic=None):
        self.env = env
        self.normalize_action = normalize_action

    def _policy(self, obs):
        dap = int(obs[0][1])
        return [self.fertilization_dic[dap] if dap in self.fertilization_dic else 0]

    def predict(self, obs, state=None, episode_start=None, deterministic=None):
        action = self._policy(obs)
        action = normalize_action((self.env.action_low, self.env.action_high), action)

        return np.array([action], dtype=np.float32), obs


# evaluation and plotting functions
def evaluate(agent, n_episodes=10):
    # Create eval env
    eval_args = {
        'mode': 'fertilization',
        'seed': 456,
        'random_weather': True,
    }
    env = Monitor(GymDssatWrapper(gym.make('GymDssatPdi-v0', **eval_args)))

    returns, _ = evaluate_policy(
        agent, env, n_eval_episodes=n_episodes, return_episode_rewards=True)

    env.close()

    return returns

def plot_results(labels, returns):
    data_dict = {}
    for label, data in zip(labels, returns):
        data_dict[label] = data
    df = pd.DataFrame(data_dict)

    ax = sns.boxplot(data=df)
    ax.set_xlabel("policy")
    ax.set_ylabel("evaluation output")
    plt.savefig(f'results_sb3_{args.cultivar}_{args.random_weather}_{args.mode}_{args.seed}.pdf')
    print("\nThe result is saved in the current working directory as 'results_sb3.pdf'\n")
    plt.show()

# evaluate agents
null_agent = NullAgent(env)
print('Evaluating Null agent...')
null_returns = evaluate(null_agent,n_episodes=400)
print('Done')

agent_returns_dict = {}
for agent_name, agent in agent_name_mapping.items():
    print(f'Evaluating {agent_name} agent...')
    agent_returns = evaluate(agent,n_episodes=400)
    agent_returns_dict[agent_name] = agent_returns
    print('Done')

expert_agent = ExpertAgent(env)
print('Evaluating Expert agent...')
expert_returns = evaluate(expert_agent,n_episodes=400)
print('Done')

# display results
labels = ['null', 'expert'] + agent_names
returns = [null_returns, expert_returns] + [agent_returns_dict[agent_name] for agent_name in agent_names]
plot_results(labels, returns)

with open(f"eval_output_{args.cultivar}_{args.random_weather}_{args.mode}_{args.seed}.txt",'w') as f:
    f.write("Null Agent : "+str(null_returns)+"\r\n")
    f.write("Expert Agent : "+str(expert_returns)+"\r\n")
    for agent_name in agent_names:
        f.write(f"{agent_name} Agent : "+str(agent_returns_dict[agent_name])+"\r\n")


# env.render(type='ts',  # time series mode
#         feature_name_1='nstres',  # mandatory first raw state variable, here the nitrogen stress factor (unitless)
#         feature_name_2='grnwt')  # optional second raw state variable, here the grain weight (kg/ha)

# Cleanup
env.close()