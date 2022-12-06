from utils.torch_utils import Tensor, Variable
import gym
from utils.models import *
from utils.torch_utils import ValueFunctionWrapper


def reward_function(cos1, sin1, cos2, sin2):
    if sin1 * sin2 - cos1 * cos2 - cos1 > -0.5:
        return 1.0
    else:
        return 0.0


def constrain_I(theta1_dot, action, penality):
    if theta1_dot < penality and action != 0:
        return -1
    else:
        return 0


def constrain_II(theta2_dot, action, penality2):
    if theta2_dot < penality2 and action != 0:
        return -1
    else:
        return 0


def sample_action_from_policy(observation, policy_model):
    observation_tensor = Tensor(observation).unsqueeze(0)
    probabilities = policy_model(Variable(observation_tensor, requires_grad=True))
    action = probabilities.multinomial(1)
    return action, probabilities


class AcrobotEnvCMDP:
    def __init__(self, length,
                 limits, gamma):

        self.env = gym.make("Acrobot-v1")
        self.length = length
        self.penalties = [0, 0]
        self.dim = 2

        self.limits = limits
        self.gamma = gamma

    def samplePaths(self, policy, episodes):
        episodes_so_far = 0
        paths = []
        total_entropy = 0

        while episodes_so_far < episodes:
            episodes_so_far += 1
            observations, actions, rewards, action_distributions = [], [], [], []
            costs = [[], []]
            observation = self.env.reset()
            length_so_far = 0
            done = False
            while length_so_far < self.length:
                if done:
                    observation = self.env.reset()

                observations.append(observation)
                action, action_dist = sample_action_from_policy(observation, policy)
                actions.append(action)
                action_distributions.append(action_dist)
                total_entropy += -(action_dist * action_dist.log()).sum()

                reward = reward_function(observation[0], observation[1], observation[2], observation[3])
                cost = constrain_I(observation[4], action, self.penalties[0])
                cost2 = constrain_II(observation[5], action, self.penalties[1])

                rewards.append(reward)
                costs[0].append(cost)
                costs[1].append(cost2)

                observation, _, done, _ = self.env.step(action[0, 0].item())
                length_so_far += 1

            path = {"observations": observations,
                    "actions": actions,
                    "rewards": rewards,
                    "costs": costs,
                    "action_distributions": action_distributions}
            paths.append(path)

        return paths, total_entropy

    def defaultPolicyModel(self):
        return DQNSoftmax(6, 3)

    def defaultValueModel(self):
        return DQNRegressor(6)
