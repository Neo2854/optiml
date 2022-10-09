import numpy as np

def Adam(deriv, x0, betas = (0.99, 0.999),eps = 1e-6, lr = 0.01, tol = 0.00001, max_iters = 10000, verbose = 0):
    m,v = 0,0

    for iter in range(max_iters):
        grad = deriv(x0)

        m = betas[0]*m + (1 - betas[0])*grad
        v = betas[1]*v + (1 - betas[1])*(grad**2)

        m_cap = m/(1 - betas[0]**(iter+1))
        v_cap = v/(1 - betas[1]**(iter+1))

        curr_update = lr*m_cap/(v_cap**0.5 + eps)

        x0 = x0 - lr*m_cap/(v_cap**0.5 + eps)

        if curr_update < tol:
            break

    return x0