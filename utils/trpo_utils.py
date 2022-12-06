import torch
from utils import math_utils
from utils.torch_utils import Tensor, Variable
from torch.nn.utils.convert_parameters import vector_to_parameters, parameters_to_vector
import numpy as np
import copy


def flatten(l):
    return [item for sublist in l for item in sublist]


def sample_trajectories(cmdp, policy, gamma, episodes=10):
    paths, entropy = cmdp.samplePaths(policy, episodes)

    observations = flatten([path["observations"] for path in paths])
    discounted_rewards = flatten([math_utils.discount(path["rewards"], gamma) for path in paths])
    discounted_rewards_sum = sum(
        [math_utils.discount(path["rewards"], gamma)[0] for path in paths]) / episodes
    total_reward = sum(flatten([path["rewards"] for path in paths])) / episodes

    total_costs = []
    discounted_costs_sum_arr = []
    discounted_costs_arr = []
    for i in range(cmdp.dim):
        discounted_costs = flatten([math_utils.discount(path["costs"][i], gamma) for path in paths])
        discounted_costs_sum = sum(
            [math_utils.discount(path["costs"][i], gamma)[0] for path in paths]) / episodes

        total_cost = sum(flatten([path["costs"][i] for path in paths])) / episodes
        total_costs.append(total_cost)

        discounted_costs_arr.append(discounted_costs)
        discounted_costs_sum_arr.append(discounted_costs_sum)

    actions = flatten([path["actions"] for path in paths])
    action_dists = flatten([path["action_distributions"] for path in paths])
    entropy = entropy / len(actions)

    return observations, \
        np.asarray(discounted_rewards), total_reward, discounted_rewards_sum,\
        np.asarray(discounted_costs_arr), np.asarray(total_costs), np.asarray(discounted_costs_sum_arr),\
        actions, action_dists, entropy



def mean_kl_divergence(model, policy_model, observations):
    observations_tensor = torch.cat(
        [Variable(Tensor(observation)).unsqueeze(0) for observation in observations])
    actprob = model(observations_tensor).detach() + 1e-8
    old_actprob = policy_model(observations_tensor)
    return torch.sum(old_actprob * torch.log(old_actprob / actprob), 1).mean()


def hessian_vector_product(vector, policy_model, observations, cg_damping):
    policy_model.zero_grad()
    mean_kl_div = mean_kl_divergence(policy_model, policy_model, observations)
    kl_grad = torch.autograd.grad(
        mean_kl_div, policy_model.parameters(), create_graph=True)
    kl_grad_vector = torch.cat([grad.view(-1) for grad in kl_grad])
    grad_vector_product = torch.sum(kl_grad_vector * vector)
    grad_grad = torch.autograd.grad(
        grad_vector_product, policy_model.parameters())
    fisher_vector_product = torch.cat(
        [grad.contiguous().view(-1) for grad in grad_grad]).data
    return fisher_vector_product + (cg_damping * vector.data)


def conjugate_gradient(policy_model, observations, cg_damping, b, cg_iters, residual_tol):
    p = b.clone().data
    r = b.clone().data
    x = np.zeros_like(b.data.cpu().numpy())
    rdotr = r.double().dot(r.double())
    for _ in range(cg_iters):
        z = hessian_vector_product(Variable(p), policy_model, observations, cg_damping).squeeze(0)
        v = rdotr / p.double().dot(z.double())
        # x += v * p.cpu().numpy()
        x += v.cpu().numpy() * p.cpu().numpy()  # change II
        r -= v * z
        newrdotr = r.double().dot(r.double())
        mu = newrdotr / rdotr
        p = r + mu * p
        rdotr = newrdotr
        if rdotr < residual_tol:
            break
    return x


def surrogate_loss(theta, policy_model, observations, actions, advantage):
    new_model = copy.deepcopy(policy_model)
    vector_to_parameters(theta, new_model.parameters())
    observations_tensor = torch.cat(
        [Variable(Tensor(observation)).unsqueeze(0) for observation in observations])
    prob_new = new_model(observations_tensor).gather(
        1, torch.cat(actions)).data
    prob_old = policy_model(observations_tensor).gather(
        1, torch.cat(actions)).data + 1e-8
    return -torch.mean((prob_new / prob_old) * advantage)


def linesearch(x, policy_model, observations, actions, advantage, fullstep, expected_improve_rate):
    accept_ratio = .1
    max_backtracks = 10
    fval = surrogate_loss(x, policy_model, observations, actions, advantage)
    for (_n_backtracks, stepfrac) in enumerate(.5 ** np.arange(max_backtracks)):
        print("Search number {}...".format(_n_backtracks + 1))
        xnew = x.data.cpu().numpy() + stepfrac * fullstep
        newfval = surrogate_loss(Variable(torch.from_numpy(xnew)), policy_model, observations, actions, advantage)
        actual_improve = fval - newfval
        expected_improve = expected_improve_rate * stepfrac
        ratio = actual_improve / expected_improve
        if ratio > accept_ratio and actual_improve > 0:
            return Variable(torch.from_numpy(xnew)), stepfrac
    return x, stepfrac


def normalize(advantage, cost_advantage, lagrange):
    normal_advantage = advantage
    for i in range(len(cost_advantage)):
        normal_advantage += cost_advantage[i] * lagrange[i]
    normal_advantage = (normal_advantage - normal_advantage.mean()) / (normal_advantage.std() + 1e-8)
    return normal_advantage
