import logging
import numpy as np


logging.basicConfig(level=logging.INFO, format='%(message)s')


def get_H_inv(A, b, x):
    d = A.shape[1]
    H = np.zeros((d, d))
    for i in range(A.shape[0]):
        a = A[i:i+1].T
        H += a @ a.T / (a.T @ x - b[i])**2
    try:
        H_inv = np.linalg.inv(H)
    except:
        H_inv = np.linalg.pinv(H)
    return H_inv


def get_sigmas(A, b, x, H_inv):
    sigmas = np.zeros(A.shape[0])
    for i in range(A.shape[0]):
        a = A[i:i+1].T
        sigmas[i] = a.T @ H_inv @ a / (a.T @ x - b[i])**2
    return sigmas


def get_Q_inv(sigmas, A, b, x):
    d = A.shape[1]
    Q = np.zeros((d, d))
    for i in range(A.shape[0]):
        a = A[i:i+1].T
        Q += sigmas[i] * a @ a.T / (a.T @ x - b[i])**2
    try:
        Q_inv = np.linalg.inv(Q)
    except:
        Q_inv = np.linalg.pinv(Q)
    return Q_inv


def get_dV(sigmas, A, b, x):
    dV = np.zeros((A.shape[1], 1))
    for i in range(A.shape[0]):
        a = A[i:i+1].T
        dV -= sigmas[i] * a / (a.T @ x - b[i])
    return dV


def get_vol_center(A, b, x, n_steps, stepsize):
    for step in range(n_steps):
        H_inv = get_H_inv(A, b, x)
        sigmas = get_sigmas(A, b, x, H_inv)
        Q_inv = get_Q_inv(sigmas, A, b, x)
        dV = get_dV(sigmas, A, b, x)
        x = x - stepsize * Q_inv @ dV
    return x


def get_beta(c, x, eps, eta, H_inv):
    squared = 2 * c.T @ H_inv @ c / np.sqrt(eps * eta)
    beta = x.T @ c - np.sqrt(squared)
    return beta


def get_beta_t(c, x, eps, eta, H_inv):
    squared = 2 * c @ H_inv @ c.T / np.sqrt(eps * eta)
    beta = x.T @ c.T - np.sqrt(squared)
    return beta


def add_row(A, b, c, beta):
    A = np.vstack((A, c.T))
    b = np.append(b, beta)
    return A, b


def add_row_t(A, b, c, beta):
    A = np.vstack((A, c))
    b = np.append(b, beta)
    return A, b


def remove_row(A, b, i):
    A = np.delete(A, i, 0)
    b = np.delete(b, i)
    return A, b


def vaidya(A_0, b_0, x_0, eps, eta, K, oracle, newton_steps=5, stepsize=0.18, verbose=True):
    """Use Vaidya's method to minimize f(x)."""
    A_k, b_k = A_0, b_0
    x_k = x_0

    xs = [x_0.copy()]
    for k in range(K):
        if verbose and k % 20 == 0:
            print(f"k={k}")
        x_k = get_vol_center(A_k, b_k, x_k, newton_steps, stepsize)

        print('\n\n>>>>>>>>>>>>>>>>>>>>>> VAIDYA GOT TO: \n', x_k)
        print('\n')

        print('\n\n>>>>>>>>>>>>>>>>>>>>>> VAIDYA NOW HAS POLYTOPES - POLYPOPES: \nA:\n{}\nb:\n{}'.format(A_k, b_k))
        print('\n')

        H_inv = get_H_inv(A_k, b_k, x_k)
        sigmas = get_sigmas(A_k, b_k, x_k, H_inv)
        if (sigmas >= eps).all():
            c_k = oracle(x_k)
            beta_k = get_beta(-c_k, x_k, eps, eta, H_inv)
            A_k, b_k = add_row(A_k, b_k, -c_k, beta_k)
        else:
            i = sigmas.argmin()
            A_k, b_k = remove_row(A_k, b_k, i)

        xs.append(x_k.copy())

    return xs


def vaidya_for_logreg(A_0, b_0, x_0, eps, eta, oracle, get_loss, model, batches, criterion, newton_steps=5,
                      stepsize=0.18, maxiter=None):
    """Use Vaidya's method to minimize f(x)."""
    A_k, b_k = A_0, b_0
    x_k = x_0

    xs = [x_0.copy()]
    for k, batch in enumerate(batches):
        if maxiter is not None and k == maxiter:
            break
        x_k = get_vol_center(A_k, b_k, x_k, newton_steps, stepsize)
        H_inv = get_H_inv(A_k, b_k, x_k)
        sigmas = get_sigmas(A_k, b_k, x_k, H_inv)
        if (sigmas >= eps).all():
            c_k, loss = oracle(model, batch["inp"], batch["oup"], criterion, x_k.T)
            beta_k = get_beta_t(-c_k, x_k, eps, eta, H_inv)
            A_k, b_k = add_row_t(A_k, b_k, -c_k, beta_k)
        else:
            loss = get_loss(model, batch["inp"], batch["oup"], criterion)
            i = sigmas.argmin()
            A_k, b_k = remove_row(A_k, b_k, i)

        xs.append(x_k.copy())

        logging.info('Step {} Loss : {}'.format((k + 1), loss))

    return xs


def get_init_polytope(d, R):
    # Задать начальное множество A_0, b_0 для радиуса R
    A_0 = np.vstack((np.eye(d), -np.ones((1, d))))
    b_0 = -R * np.ones(d + 1)
    b_0[-1] *= d
    return A_0, b_0


def get_init_polytope_square(d, R):
    A_0 = np.vstack((np.eye(d), -np.eye(d)))
    b_0 = -R * np.ones(2 * d)
    return A_0, b_0
