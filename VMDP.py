import collections
from utils.torch_utils import ValueFunctionWrapper
from utils.trpo_utils import *
import numpy as np

from vaidya.vaidya import vaidya, get_init_polytope

from step import step_estimate_gradient



def vaidya_mdp(cmdp,
               policy_model, value_model, cost_models,
               n_inner, n_outer, episodes, use_discounted_reward=False):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device: ', device)

    ## Hyperparameter
    value_function_lr = 1
    max_kl = 0.003
    cg_damping = 0.001
    cg_iters = 10
    residual_tol = 1e-10
    ent_coeff = 0.001
    batch_size = 5100
    max_kl = 0.01
    mu = 0.01
    value_lr = 1

    ## Initial neural network
    ## already initialized

    # Statistics across oracle calls
    rewards = []
    disc_rewards = []
    costs = []
    disc_costs = []

    lagranges = []
    models = []
    oracle_calls_cnt = 0

    def oracle(lagrange):

        nonlocal oracle_calls_cnt

        print('Oracle call number {} to lambda =\n {}\n'.format(oracle_calls_cnt, lagrange))
        oracle_calls_cnt += 1

        for i in range(cmdp.dim):
            if lagrange[i][0] < 0:
                return np.where(lagrange.reshape(lagrange.shape[0]) < 0, -1, 0)

        gradient_lagrange = np.zeros((cmdp.dim, 1))

        for inner_iteration in range(n_inner):
            print('\n\nINNER STEP {}\n'.format(inner_iteration))
            total_reward, disc_reward,\
            violations, disc_violations, \
            gradient_lagrange = step_estimate_gradient(lagrange,
                                     cmdp,
                                     episodes,
                                     policy_model, value_model, cost_models,
                                     value_lr, batch_size, max_kl, cg_iters, residual_tol, cg_damping, ent_coeff, mu,
                                     use_discounted_reward)

            print('GRADIENT: ', gradient_lagrange)

            # gather statistics

            rewards.append(total_reward)
            costs.append(-1 * violations)

            disc_rewards.append(disc_reward)
            disc_costs.append(-1 * disc_violations)

            lagranges.append(lagrange)

            models.append(copy.deepcopy(policy_model))

        return gradient_lagrange


    # Setting Vaidya parameters
    m = cmdp.dim  # Problem dimension = m
    lam_0 = np.zeros((m, 1))  # First vol center approximation
    R = 1  # Optimize on set: ||lambda||_2 <= R.
    A_0, b_0 = get_init_polytope(m, R)  # Generate initial polytope
    b_0 -= (m - 1) / (m + 1) * R
    b_0[-1] += (m - 1) / (m + 1) * R
    b_0[-1] += (m - 1) / (m + 1) * R * m
    K = n_outer  # Number of iterations

    eta = 1000
    factor = 1e-4

    eps = eta * factor

    print('VAIDYA PARAMETERS')
    print('INNNER_ITER: {}\n' \
          'OUTER ITER: {}\n' \
          'eta: {}\n' \
          'factor: {}\n' \
          'R: {}\n' \
          'A_0: {}\n' \
          'b_0: {}\n' \
          'lam_0: {}\n' \
          'eps: {}\n' \
          .format(n_inner, n_outer, eta, factor, R, A_0, b_0, lam_0, eps))

    vaidya(A_0, b_0, lam_0, eps, eta, K, oracle, newton_steps=10, stepsize=0.18, verbose=False)

    result = {'rewards': rewards,
            'discrewards': disc_rewards}

    for i in range(m):
        result['lagrange{}'.format(i + 1)] = [lagranges[it][i] for it in range(len(costs))]
        result['costs{}'.format(i + 1)] = [costs[it][i] for it in range(len(costs))]
        result['disccosts{}'.format(i + 1)] = [disc_costs[it][i] for it in range(len(disc_costs))]

    return result
