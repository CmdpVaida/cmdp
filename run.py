import torch
import numpy as np
import random

from utils.pretrain import pretrain, pretrained_load
from utils import experiment
from utils.experiment import plotsave

from CMDP_envs.AcrobotEnv import AcrobotEnvCMDP

from VMDP import vaidya_mdp
from AR_CPO import arcpo


torch.manual_seed(1791791791)
np.random.seed(1791791791)
random.seed(1791791791)

repeat_times = 2
OUTER_STEPS = 10
INNER_STEPS = 5
TOTAL_STEPS = OUTER_STEPS * INNER_STEPS

PRETRAIN_STEPS = 0

cmdp = AcrobotEnvCMDP(length=500, limits=[5,5], gamma=0.98)

experiment_dir = 'results'

#===============Pretrain and Save==============================================#


SAVE_PATH = experiment_dir + '/model_pretrained_weights'
experiment.mkdir(SAVE_PATH)

for i in range(repeat_times):
    curr_path = SAVE_PATH + "/iter_" + str(i)
    experiment.mkdir(curr_path)
    pretrain(cmdp, PRETRAIN_STEPS, curr_path)

#===============Conduct experiments and Save results==============================================#


# VMDP = experiment.ExperimentResult('VMDP')
#
# for i in range(repeat_times):
#     curr_path = SAVE_PATH + "/iter_" + str(i)
#     policy_model, value_model, cost_models = pretrained_load(cmdp, curr_path)
#     VMDP.append(vaidya_mdp(cmdp,
#                 policy_model, value_model, cost_models,
#                 n_inner=INNER_STEPS, n_outer=OUTER_STEPS, episodes=10, use_discounted_reward=True))
#
# VMDP.save(experiment_dir + '/vmdp_results')
#
#
# ARCPO = experiment.ExperimentResult('ARCPO')
# for i in range(repeat_times):
#     curr_path = SAVE_PATH + "/iter_" + str(i)
#     policy_model, value_model, cost_models = pretrained_load(cmdp, curr_path)
#     ARCPO.append(arcpo(cmdp,
#                  policy_model, value_model, cost_models,
#                  n_inner=INNER_STEPS, n_outer=OUTER_STEPS, episodes=10, use_discounted_reward=True))
# ARCPO.save(experiment_dir + '/arcpo_results')


