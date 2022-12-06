from VMDP import step_estimate_gradient
import torch


def pretrain(cmdp, n_iters, save_path):
    """
    Train default networks for $cmdp
    :param cmdp: 
    :param n_iters: 
    :param save_path: 
    :return: None
    """

    episodes = 10
    cg_damping = 0.001
    cg_iters = 10
    residual_tol = 1e-10
    ent_coeff = 0.00
    batch_size = 5100
    max_kl = 0.01
    value_function_lr = 1

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    policy_model = cmdp.defaultPolicyModel()
    value_model = cmdp.defaultValueModel()
    cost_models = [cmdp.defaultValueModel() for _ in range(cmdp.dim)]

    policy_model = policy_model.to(device)
    value_model = value_model.to(device)
    cost_models = [cost_model.to(device) for cost_model in cost_models]


    for inner_iteration in range(n_iters):
        print('\n\nPRETRAIN STEP {}\n'.format(inner_iteration))
        step_estimate_gradient([0] * cmdp.dim,
                                cmdp,
                                episodes,
                                policy_model, value_model, cost_models,
                                batch_size, max_kl, cg_iters, residual_tol, cg_damping, ent_coeff, 0,
                                False)

    torch.save(policy_model.state_dict(), save_path + "/policy_model.pck")
    torch.save(value_model.state_dict(), save_path + "/value_model.pck")
    for i in range(len(cost_models)):
        torch.save(cost_models[i].state_dict(), save_path + "/cost_model_{}.pck".format(i))


def pretrained_load(cmdp, path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    policy_model = cmdp.defaultPolicyModel()
    value_model = cmdp.defaultValueModel()
    cost_models = [cmdp.defaultValueModel() for _ in range(cmdp.dim)]

    policy_model = policy_model.to(device)
    value_model = value_model.to(device)
    cost_models = [cost_model.to(device) for cost_model in cost_models]

    policy_model.load_state_dict(torch.load(path + "/policy_model.pck"))
    value_model.load_state_dict(torch.load(path + "/value_model.pck"))
    for i in range(len(cost_models)):
        cost_models[i].load_state_dict(torch.load(path + "/cost_model_{}.pck".format(i)))

    return policy_model, value_model, cost_models
