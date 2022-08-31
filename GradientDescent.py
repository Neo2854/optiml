import numpy as np

def GradientDescent(deriv, x0, lr = 0.01, tol = 0.00001, max_iters = 10000, verbose = 0):
    curr_tol = tol * 1000
    dir = np.zeros(x0.shape)

    while curr_tol >= tol:
        grad = deriv(x0)
        dir = -1*grad

        prev_x = x0
        x0 = x0 + lr*dir

        curr_tol = np.linalg.norm(x0 - prev_x)

        max_iters -= 1

        if verbose:
            print("Iterations rem:", max_iters, ", x =", x0, ", tol =", curr_tol)

        if max_iters <= 0:
            break

    return x0