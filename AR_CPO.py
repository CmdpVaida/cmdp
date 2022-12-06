import torch
import numpy as np
from step import step_estimate_gradient


def arcpo(cmdp,
          policy_model, value_model, cost_models, n_inner, n_outer, episodes,
          use_discounted_reward=False):

    m = cmdp.dim

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
    stopping_time = 25
    value_lr = 1
    eta = 0.0003
    s = 1

    ## Initial neural network
    ## already initialized

    # Statistics across oracle calls
    rewards = []
    disc_rewards = []
    costs = []
    disc_costs = []

    lagranges = []
    upper_lagranges = []
    lower_lagranges = []

    models = []

    upper_lagrange = np.zeros(m)
    lower_lagrange = np.zeros(m)
    lagrange = np.zeros(m)

    for iteration in range(n_outer):

        if iteration < stopping_time:
            q = 2.0 * s / (float(iteration) + 2)
        else:
            q = 2.0 * s / float(stopping_time + 1)

        # qs.append(q)

        lower_lagrange = (1 - q) * upper_lagrange + q * lagrange

        for inner_iteration in range(n_inner):

            reward, disc_reward, violations, disc_violations, \
            gradient_lagrange = step_estimate_gradient(lower_lagrange,
                                                     cmdp,
                                                     episodes,
                                                     policy_model, value_model, cost_models,
                                                     value_lr, batch_size, max_kl, cg_iters, residual_tol, cg_damping, ent_coeff, mu,
                                                     use_discounted_reward)

            rewards.append(reward)
            costs.append(-1 * violations)

            disc_rewards.append(disc_reward)
            disc_costs.append(-1 * disc_violations)

            running_iteration = iteration * n_inner + inner_iteration

            lower_lagranges.append(lower_lagrange)
            lagranges.append(lagrange)
            upper_lagranges.append(upper_lagrange)

            if (running_iteration + 1) % 50 == 0:
                print("step:", running_iteration + 1, "test result:", reward, "test violations:", -1 * violations)

            print('lagrange1, lagrange2 is!', lagrange)
            print('lower_lagrange1, lower_lagrange2 is', lower_lagrange)
            print('DEBUG in grads', gradient_lagrange)

        lagrange = np.max(lagrange - eta * gradient_lagrange, 0)

        if iteration < stopping_time:
            alpha = 2.0 * s / (float(iteration) + 2)
        else:
            alpha = 2.0 * s / float(stopping_time + 1)

        # alphas.append(alpha)
        upper_lagrange = (1 - alpha) * upper_lagrange + alpha * lagrange

    result = {'rewards': rewards,
              'discrewards': disc_rewards}

    for i in range(m):
        result['lagrange{}'.format(i + 1)] = [lagranges[it][i] for it in range(len(costs))]
        result['costs{}'.format(i + 1)] = [costs[it][i] for it in range(len(costs))]
        result['disccosts{}'.format(i + 1)] = [disc_costs[it][i] for it in range(len(disc_costs))]

    return result
