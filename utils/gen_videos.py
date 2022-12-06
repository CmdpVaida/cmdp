import gym
from gym.wrappers import Monitor
import torch
from before_refactor.VMDP import sample_action_from_policy
from before_refactor.VMDP import DQNSoftmax


def wrap_env(env, video_callable=None):
    env = Monitor(env, './video', force=True, video_callable=video_callable)
    return env


policy_model = DQNSoftmax(6, 3)
policy_model.load_state_dict(torch.load('VMDP_result/results/models_0_109.txt'))
policy_model.to('cpu')


env = gym.make('Acrobot-v1')
env.seed(0)
env = wrap_env(env, video_callable=lambda episode_id: True)

for num_episode in range(1):
    state = env.reset()
    score = 0
    done = False
    # Go on until the pole falls off or the score reach -500
    while not done and score > -500:
        # Choose a random action
        action = sample_action_from_policy(state, policy_model)[0]
        next_state, reward, done, info = env.step(action)
        # Visually render the environment
        # Update the final score (-1 for each step)
        score += reward
        state = next_state
        # Check if the episode ended (the pole fell down)
    print(f"EPISODE {num_episode + 1} - FINAL SCORE: {score}")

