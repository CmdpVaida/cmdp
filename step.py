import collections
from utils.torch_utils import ValueFunctionWrapper
from utils.trpo_utils import *
import numpy as np


def step_estimate_gradient(lagrange,
         cmdp,
         episodes,
         policy_model, value_model, cost_models,
         value_lr, batch_size, max_kl, cg_iters, residual_tol, cg_damping, ent_coeff, mu,
         use_discounted_reward=False):

    policy_model = ValueFunctionWrapper(policy_model, value_lr)
    value_model = ValueFunctionWrapper(value_model, value_lr)
    cost_models = [ValueFunctionWrapper(model, value_lr) for model in cost_models]

    lagrange = np.array(lagrange)
    m = lagrange.shape[0]
    gamma = cmdp.gamma
    limits = cmdp.limits

    # Generate $episodes rollouts
    all_observations, all_discounted_rewards, total_reward, \
        discounted_rewards_sum, all_discounted_costs, total_costs, discounted_costs_sum,\
        all_actions, all_action_dists, entropy = sample_trajectories(cmdp, policy_model, gamma, episodes)

    num_batches = len(all_actions) // batch_size + 1
    for batch_num in range(num_batches):
        print("Processing batch number {}".format(batch_num + 1))
        observations = all_observations[batch_num * batch_size:(batch_num + 1) * batch_size]
        discounted_rewards = all_discounted_rewards[batch_num * batch_size:(batch_num + 1) * batch_size]
        discounted_costs = all_discounted_costs[:][batch_num * batch_size:(batch_num + 1) * batch_size]
        actions = all_actions[batch_num * batch_size:(batch_num + 1) * batch_size]
        action_dists = all_action_dists[batch_num * batch_size:(batch_num + 1) * batch_size]

        # Calculate the advantage of each step by taking the actual discounted rewards seen
        # and subtracting the estimated value of each state
        baseline = value_model.predict(observations).data
        discounted_rewards_tensor = Tensor(discounted_rewards).unsqueeze(1)
        advantage = discounted_rewards_tensor - baseline

        discounted_costs_tensor = Tensor(discounted_costs).unsqueeze(2)
        cost_advantages = []
        for i in range(m):
            cost_baseline = cost_models[i].predict(observations).data
            cost_advantages.append(discounted_costs_tensor[i] - cost_baseline)

        # Normalize the advantage
        lagrange_advantage = normalize(advantage, cost_advantages, lagrange)

        new_p = torch.cat(action_dists).gather(1, torch.cat(actions))
        old_p = new_p.detach() + 1e-8
        prob_ratio = new_p / old_p

        lagrange_surrogate_loss = - torch.mean(prob_ratio * Variable(lagrange_advantage)) - (ent_coeff * entropy)

        # Calculate the gradient of the surrogate loss
        policy_model.zero_grad()
        lagrange_surrogate_loss.backward(retain_graph=True)

        policy_gradient = parameters_to_vector([v.grad for v in policy_model.parameters()]).squeeze(0)

        if policy_gradient.nonzero().size()[0]:
            # Use conjugate gradient algorithm to determine the step direction in theta space
            step_direction = conjugate_gradient(policy_model, observations, cg_damping, -policy_gradient, cg_iters,
                                                residual_tol)
            step_direction_variable = Variable(torch.from_numpy(step_direction))

            # Do line search to determine the stepsize of theta in the direction of step_direction
            shs = .5 * step_direction.dot(
                hessian_vector_product(step_direction_variable, policy_model, observations, cg_damping).cpu().numpy().T)
            lm = np.sqrt(shs / max_kl)
            fullstep = step_direction / lm
            gdotstepdir = -policy_gradient.dot(step_direction_variable).data.item()  # change III

            theta, stepfrac = linesearch(parameters_to_vector(policy_model.parameters()), policy_model, observations,
                                         actions,
                                         lagrange_advantage, fullstep, gdotstepdir / lm)

            ev_before = math_utils.explained_variance_1d(baseline.squeeze(1).cpu().numpy(), discounted_rewards)

            value_model.zero_grad()
            for model in cost_models:
                model.zero_grad()
                model.zero_grad()

            value_model.fit(observations, Variable(discounted_rewards_tensor))
            for i in range(len(cost_models)):
                cost_models[i].fit(observations, Variable(discounted_costs_tensor[i]))

            ev_after = math_utils.explained_variance_1d(
                value_model.predict(observations).data.squeeze(1).cpu().numpy(), discounted_rewards)

            old_model = copy.deepcopy(policy_model)
            old_model.load_state_dict(policy_model.state_dict())
            if any(np.isnan(theta.data.cpu().numpy())):
                print("NaN detected. Skipping update...")
            else:
                vector_to_parameters(theta, policy_model.parameters())

            kl_old_new = mean_kl_divergence(old_model, policy_model, observations)
            diagnostics = collections.OrderedDict(
                [('Total Reward', total_reward), ('Total Cost', -1 * total_costs),
                 ('Discounted Reward', discounted_rewards_sum), ('Discounted Cost', -1 * discounted_costs_sum),
                 ('KL Old New', kl_old_new.data.item()), ('Entropy', entropy.data.item()), ('EV Before', ev_before),
                 ('EV After', ev_after)])
            for key, value in diagnostics.items():
                print("{}: {}".format(key, value))

        else:
            print("Policy gradient is 0. Skipping update...")

    limits_arr = np.array(limits).reshape((len(limits), 1))
    if use_discounted_reward:
        gradient_lagrange = discounted_costs_sum.reshape((m, 1)) - limits_arr + mu * lagrange
    else:
        gradient_lagrange = total_costs.reshape((m, 1)) - limits_arr + mu * lagrange

    return total_reward, discounted_rewards_sum, \
        total_costs, discounted_costs_sum, \
        gradient_lagrange


def step_pdo(lagrange,
             cmdp,
             episodes,
             policy_model, value_model, cost_models,
             value_lr, batch_size, max_kl, cg_iters, residual_tol, cg_damping, ent_coeff, eta,
             use_discounted_reward=False):

    policy_model = ValueFunctionWrapper(policy_model, value_lr)
    value_model = ValueFunctionWrapper(value_model, value_lr)
    cost_models = [ValueFunctionWrapper(model, value_lr) for model in cost_models]

    m = lagrange.shape[0]
    lagrange = np.array(lagrange).reshape(m, 1)
    limits = np.array(cmdp.limits).reshape(m, 1)
    gamma = cmdp.gamma

    # Generate $episodes rollouts
    all_observations, all_discounted_rewards, total_reward, \
        discounted_rewards_sum, all_discounted_costs, total_costs, discounted_costs_sum,\
        all_actions, all_action_dists, entropy = sample_trajectories(cmdp, policy_model, gamma, episodes)

    num_batches = len(all_actions) // batch_size + 1
    for batch_num in range(num_batches):
        print("Processing batch number {}".format(batch_num+1))
        observations = all_observations[batch_num * batch_size:(batch_num+1)*batch_size]
        discounted_rewards = all_discounted_rewards[batch_num*batch_size:(batch_num+1)*batch_size]
        discounted_costs = all_discounted_costs[:, batch_num * batch_size:(batch_num + 1) * batch_size]
        actions = all_actions[batch_num * batch_size:(batch_num+1)*batch_size]
        action_dists = all_action_dists[batch_num * batch_size:(batch_num+1)*batch_size]

        # Calculate the advantage of each step by taking the actual discounted rewards seen
        # and subtracting the estimated value of each state
        baseline = value_model.predict(observations).data
        discounted_rewards_tensor = Tensor(discounted_rewards).unsqueeze(1)
        advantage = discounted_rewards_tensor - baseline

        discounted_costs_tensor = Tensor(discounted_costs).unsqueeze(2)
        cost_advantages = []
        for i in range(m):
            cost_baseline = cost_models[i].predict(observations).data
            cost_advantages.append(discounted_costs_tensor[i] - cost_baseline)

        # Normalize the advantage
        lagrange_advantage = normalize(advantage, cost_advantages, lagrange)

        # Calculate the surrogate loss as the elementwise product of the advantage and the probability ratio of actions taken
        new_p = torch.cat(action_dists).gather(1, torch.cat(actions))
        old_p = new_p.detach() + 1e-8
        prob_ratio = new_p / old_p

        lagrange_surrogate_loss = - torch.mean(prob_ratio * Variable(lagrange_advantage)) - (ent_coeff * entropy)

        # Calculate the gradient of the surrogate loss
        policy_model.zero_grad()
        lagrange_surrogate_loss.backward(retain_graph=True)

        policy_gradient = parameters_to_vector([v.grad for v in policy_model.parameters()]).squeeze(0)

        if policy_gradient.nonzero().size()[0]:
            # Use conjugate gradient algorithm to determine the step direction in theta space
            step_direction = conjugate_gradient(policy_model, observations, cg_damping, -policy_gradient, cg_iters, residual_tol)
            step_direction_variable = Variable(torch.from_numpy(step_direction))

            # Do line search to determine the stepsize of theta in the direction of step_direction
            shs = .5 * step_direction.dot(hessian_vector_product(step_direction_variable, policy_model, observations, cg_damping).cpu().numpy().T)
            lm = np.sqrt(shs / max_kl)
            fullstep = step_direction / lm
            gdotstepdir = -policy_gradient.dot(step_direction_variable).data.item()

            theta, stepfrac = linesearch(parameters_to_vector(policy_model.parameters()), policy_model, observations, actions,
                             lagrange_advantage, fullstep, gdotstepdir / lm)

            if use_discounted_reward:
                updated = lagrange + eta * (limits - discounted_costs.reshape(m, 1))
                lagrange = np.maximum(updated, np.zeros_like(updated))
            else:
                updated = lagrange + eta * (limits - total_costs.reshape(m, 1))
                lagrange = np.maximum(updated, np.zeros_like(updated))

            # Fit the estimated value function to the actual observed discounted rewards

            # collect EV before fit
            ev_before = math_utils.explained_variance_1d(
                value_model.predict(observations).data.squeeze(1).cpu().numpy(), discounted_rewards)

            cost_evs_before = []
            for i in range(m):
                cost_evs_before.append(math_utils.explained_variance_1d(
                    cost_models[i].predict(observations).data.squeeze(1).cpu().numpy(), discounted_costs[i]))

            # fit value/cost models

            value_model.zero_grad()
            for model in cost_models:
                model.zero_grad()

            # collect parameters for potential revert
            value_fn_params = parameters_to_vector(value_model.parameters())
            cost_fns_params = []
            for model in cost_models:
                cost_fns_params.append(parameters_to_vector(model.parameters()))

            value_model.fit(observations, Variable(discounted_rewards_tensor))
            for i in range(m):
                cost_models[i].fit(observations, Variable(discounted_costs_tensor[i]))

            # collect EV after fit

            ev_after = math_utils.explained_variance_1d(
              value_model.predict(observations).data.squeeze(1).cpu().numpy(), discounted_rewards)

            cost_evs_after = []
            for i in range(m):
                cost_evs_after.append(math_utils.explained_variance_1d(
                  cost_models[i].predict(observations).data.squeeze(1).cpu().numpy(), discounted_costs[i]))

            # If something wrong with relation of EV before and after, revert the update

            if ev_after < ev_before or np.abs(ev_after) < 1e-4:
                vector_to_parameters(value_fn_params, value_model.parameters())

            for i in range(m):
                if cost_evs_after[i] < cost_evs_before[i] or np.abs(cost_evs_after[i]) < 1e-4:
                    vector_to_parameters(cost_fns_params[i], cost_models[i].parameters())

            # Update parameters of policy model
            old_model = copy.deepcopy(policy_model)
            old_model.load_state_dict(policy_model.state_dict())
            if any(np.isnan(theta.data.cpu().numpy())):
                print("NaN detected. Skipping update...")
            else:
                vector_to_parameters(theta, policy_model.parameters())

            kl_old_new = mean_kl_divergence(old_model, policy_model, observations)
            diagnostics = collections.OrderedDict([('Total Reward', total_reward),
                                                 ('KL Old New', kl_old_new.data.item()), ('Entropy', entropy.data.item()), ('EV Before', ev_before),
                                                 ('EV After', ev_after)])
            for i in range(m):
                diagnostics['Costs {}'.format(i)] = total_costs[i]

            for key, value in diagnostics.items():
                print("{}: {}".format(key, value))

        else:
            print("Policy gradient is 0. Skipping update...")

    return total_reward, discounted_rewards_sum, \
           total_costs, discounted_costs_sum, \
           lagrange
