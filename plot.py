from utils import experiment
from utils.experiment import plotsave

experiment_dir = 'results'

VMDP = experiment.ExperimentResult('VMDP_result')
VMDP.load(experiment_dir + '/vmdp_results')

ARCPO = experiment.ExperimentResult('ARCPO')
ARCPO.load(experiment_dir + '/arcpo_results')

PDO = experiment.ExperimentResult('PDO')
PDO.load(experiment_dir + '/pdo_results')

plots_dir = experiment_dir + '/plots'
experiment.mkdir(plots_dir)


plotsave(plots_dir + '/Rewards.pdf',
         {'rewards_vmdp': VMDP.results['discrewards'],
          'reward_arcpo': ARCPO.results['discrewards'],
          'reward_pdo': PDO.results['discrewards']},
         ylim=20,
         episodes=10)

plotsave(plots_dir + '/Costs1.pdf',
         {'cost1_arcpo': ARCPO.results['disccosts1'],
          'cost1_vmdp': VMDP.results['disccosts1'],
          'cost1_pdo': PDO.results['disccosts1']},
         ylim=20,
         hor_ticks_height=5,
         episodes=10)

plotsave(plots_dir + '/Costs2.pdf',
         {'cost2_arcpo': ARCPO.results['disccosts2'],
          'cost2_vmdp': VMDP.results['disccosts2'],
          'cost2_pdo': PDO.results['disccosts2']},
         ylim=20,
         hor_ticks_height=5,
         episodes=10)