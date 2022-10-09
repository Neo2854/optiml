import numpy as np

def adagrad(deriv, x0, lr = 1e-2, fudge_factor = 1e-6, tol=1e-6, max_iter=1000):
    gti=np.zeros(x0.shape[0])

    for iter in range(max_iter):
        grad = deriv(x0)
        gti += grad**2
        adjusted_grad = grad / (fudge_factor + np.sqrt(gti))
        x0 = x0 - lr*adjusted_grad

        if lr*adjusted_grad < tol:
            break

    return x0