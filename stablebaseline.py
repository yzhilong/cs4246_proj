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
import pickle

parser = argparse.ArgumentParser(
    prog='train selected crops',
)
parser.add_argument('-c', '--cultivar', default='maize', type=str)
parser.add_argument('-s', '--seed', default=123, type=int)
parser.add_argument('-r', '--random_weather', default='True', type=str)
parser.add_argument('-m', '--mode', default='fertilization', type=str)
parser.add_argument('-n', '--num_eps', default=400, type=int)
parser.add_argument('-t', '--tsteps', default=100_000, type=int)
args = parser.parse_args()
args.random_weather = args.random_weather.lower() == 'True'.lower()
print(args)



TOTAL_TIMESTEPS = args.tsteps
N_EPISODES = args.num_eps

# helpers for action normalization
def normalize_action(action_space_limits, action):
    """Normalize the action from [low, high] to [-1, 1]"""
    output = []
    for i in range(len(action_space_limits) // 2):
        low, high = action_space_limits[2*i], action_space_limits[2*i+1]
        output.append(2.0 * ((action[i] - low) / (high - low)) - 1.0)
    return np.array(output)

def denormalize_action(action_space_limits, action):
    """Denormalize the action from [-1, 1] to [low, high]"""
    output = []
    for i in range(len(action_space_limits) // 2):
        low, high = action_space_limits[2*i], action_space_limits[2*i+1]
        output.append(low + (0.5 * (action[i] + 1.0) * (high - low)))
    return np.array(output)

# Wrapper for easy and uniform interfacing with SB3
class GymDssatWrapper(gym.Wrapper):
    def __init__(self, env):
        super(GymDssatWrapper, self).__init__(env)

        # using a normalized action space
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(2,) if env.mode == 'all' else (1,), dtype="float32")
        self.mode = env.mode

        mode_action_mapping = {
            'all': ['anfer', 'amir'],
            'fertilization': ['anfer'],
            'irrigation': ['amir']
        }

        self.action_list = mode_action_mapping[self.mode]
        self.action_bounds = self._get_action_space_bounds()

        # using a vector representation of observations to allow
        # easily using SB3 MlpPolicy
        self.observation_space = gym.spaces.Box(low=0.0,
                                                high=np.inf,
                                                shape=env.observation_dict_to_array(
                                                    env.observation).shape,
                                                dtype="float32"
                                                )
        print("Observation_space shape:", self.observation_space.shape)

        # to avoid annoying problem with Monitor when episodes end and things are None
        self.last_info = {}
        self.last_obs = None

    def _get_action_space_bounds(self):
        output = []
        for action in self.action_list:
            b = self.env.action_space[action]
            output.append(b.low)
            output.append(b.high)
        return np.array(output)

    def _format_action(self, action):
        if self.mode == 'all':
            return { 'anfer': action[0], 'amir': action[1] }
        mode_name_mapping = {
            'fertilization': 'anfer',
            'irrigation': 'amir'
        }
        return { mode_name_mapping[self.mode]: action[0] }

    def _format_observation(self, observation):
        return self.env.observation_dict_to_array(observation)

    def reset(self):
        return self._format_observation(self.env.reset())


    def step(self, action):
        # Rescale action from [-1, 1] to original action space interval
        denormalized_action = denormalize_action(self.action_bounds, action)
        formatted_action = self._format_action(denormalized_action)
        obs, reward, done, info = self.env.step(formatted_action)
        if type(reward) == list:
            reward = sum(reward)

        # handle `None`s in obs, reward, and info on done step
        if done:
            obs, reward, info = self.last_obs, 0, self.last_info
        else:
            self.last_obs = obs
            self.last_info = info

        formatted_observation = self._format_observation(obs)
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
    'gamma': 1,
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
        action = normalize_action(self.env.action_bounds, [0, 0])
        return np.array([action], dtype=np.float32), obs


class ExpertAgent:
    """
    Simple agent using policy of choosing fertilization amount based on days after planting
    """
    ALL_POLICIES = {
        'maize': {
            'fertilization_dic': {
                40: 27,
                45: 35,
                80: 54,
            },
            'irrigation_dic':  {
                6: 13,
                20: 10,
                37: 10,
                50: 13,
                54: 18,
                65: 25,
                69: 25,
                72: 13,
                75: 15,
                77: 19,
                80: 20,
                84: 20,
                91: 15,
                101: 19,
                104: 4,
                105: 25,
            }
        },
        'rice':
            {
            'fertilization_dic': {
                1: 120,
            },
            'irrigation_dic': {
                7: 27,
                10: 26,
                14: 26,
                20: 25,
                27: 28,
                31: 27,
                38: 31,
                42: 14,
                45: 11,
                49: 28,
                52: 23,
                57: 8,
                60: 17,
                63: 18,
                65: 11,
                67: 10,
                69: 8,
                71: 15,
                73: 13,
                76: 27,
                78: 16,
                81: 19,
                90: 23,
                94: 22,
                99: 24,
                101: 15,
                103: 14,
                105: 11,
                108: 19,
                111: 5,
                112: 14,
                115: 20,
                121: 15,

            }
            },
        'cotton': {
            'fertilization_dic': {
                1: 67,
                39: 22,
                66: 22,
                100: 22
            },
            'irrigation_dic': {
                1: 10,
                7: 43,
                11: 43,
                14: 34,
                18: 47,
                22: 43,
                24: 30,
                32: 73,
                38: 30,
                49: 52,
                56: 43,
                71: 60,
                73: 13,
                75: 47,
                78: 26,
                81: 26,
                85: 34,
                89: 60,
                92: 56,
                93: 13,
                94: 34,
                101: 26,
                103: 17,
                105: 34,
                107: 17,
                109: 30,
                111: 22,
                113: 17,
                115: 9,
                117: 22,
                119: 13,
                121: 22,
                123: 17,
                125: 30,
                127: 26
            }
        }

    }
    fertilization_dic = ALL_POLICIES[args.cultivar]['fertilization_dic']
    irrigation_dic = ALL_POLICIES[args.cultivar]['irrigation_dic']

    def __init__(self, env, normalize_action=False, fertilization_dic=None):
        self.env = env
        self.normalize_action = normalize_action

    def _policy(self, obs):
        dap = int(obs[0][1])
        fert = self.fertilization_dic[dap] if dap in self.fertilization_dic else 0
        irr = self.irrigation_dic[dap] if dap in self.irrigation_dic else 0
        return [fert, irr]

    def predict(self, obs, state=None, episode_start=None, deterministic=None):
        action = self._policy(obs)
        action = normalize_action(self.env.action_bounds, action)

        return np.array([action], dtype=np.float32), obs


# evaluation and plotting functions
def evaluate(agent, agent_name, n_episodes=10):
    # Create eval env
    eval_args = env_args.copy()

    # Ensures that we are not testing on train data
    eval_args['seed'] += 1

    env = Monitor(GymDssatWrapper(gym.make('GymDssatPdi-v0', **eval_args)))
    returns, _, hist = evaluate_policy(
        agent, env, n_eval_episodes=n_episodes, return_episode_rewards=True, crop_type=args.cultivar
    )
    with open(f"{agent_name}_{args.mode}_{args.cultivar}_history.pkl", 'wb') as f:
        pickle.dump(hist, f)


    # if agent_name is not None:
    #     save_env_history(env, agent_name)
    # returns, _, grain_weights = evaluate_policy(
    #     agent, env, n_eval_episodes=n_episodes, return_episode_rewards=True, crop_type=args.cultivar
    # )
    # n = len(grain_weights)
    # with open(f"{args.cultivar}_{args.random_weather}_{args.mode}_{args.seed}_{agent_name}_grainweights.txt", 'w') as f:
    #     f.write(str(grain_weights))
    # grain_weights = pd.DataFrame({f"Eps{i}": grain_weights[i*n//10:(i+1)*n//10] for i in range(n_episodes)})
    # grain_weights.to_csv(f"{agent_name}_grainweights.csv", index=False)

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
    print(f"\nThe result is saved in the current working directory as 'results_sb3_{args.cultivar}_{args.random_weather}_{args.mode}_{args.seed}.pdf'\n")
    plt.show()

def save_env_history(env, name):
    obs = env.history['observation']
    print(len(obs), type(obs))


# evaluate agents
null_agent = NullAgent(env)
print('Evaluating Null agent...')
null_returns = evaluate(null_agent,n_episodes=N_EPISODES, agent_name='null')
print('Done')

agent_returns_dict = {}
for agent_name, agent in agent_name_mapping.items():
    print(f'Evaluating {agent_name} agent...')
    agent_returns = evaluate(agent,n_episodes=N_EPISODES, agent_name=agent_name)
    agent_returns_dict[agent_name] = agent_returns
    print('Done')

expert_agent = ExpertAgent(env)
print('Evaluating Expert agent...')
expert_returns = evaluate(expert_agent,n_episodes=N_EPISODES, agent_name='expert')
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