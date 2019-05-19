

import numpy as np

def func(x):
    return x[0] ** 3 + 8 * x[1] ** 3 - 6 * x[0] * x[1] + 5

def jacb(x):

    return np.array([3 * x[0] ** 2 - 6 * x[1], 24 * x[1] ** 2 - 6 * x[0]])

def hess(x):
    return np.array([[6 * x[0], -6], [-6, 48 * x[1]]])


def dogleg(gk, Bk, trust_radius):

    """
    Dogleg trust region algorithm.
                  / tau . pU            0 <= tau <= 1,
        p(tau) = <
                  \ pU + (tau - 1)(pB - pU),    1 <= tau <= 2.
    where:
        - tau is in [0, 2]
        - pU is the unconstrained minimiser along the steepest descent direction.
        - pB is the full step.
    pU is defined by the formula::
                gT.g
        pU = - ------ g
               gT.B.g
    and pB by the formula::
        pB = - B^(-1).g
    If the full step is within the trust region it is taken.
    Otherwise the point at which the dogleg trajectory intersects the trust region is taken.
    This point can be found by solving the scalar quadratic equation:
        ||pU + (tau - 1)(pB - pU)||^2 = delta^2
    """

    pB = -np.linalg.inv(Bk).dot(gk)
    norm_pB = np.sqrt(np.dot(pB, pB))
    if(norm_pB < trust_radius):
        return pB

    pU = - (np.dot(gk, gk) / np.dot(gk, np.dot(Bk, gk))) * gk

    norm_pU = np.sqrt(np.dot(pU, pU))

    # caucy point is outside of the trust-region
    if (norm_pU > trust_radius):
        return trust_radius * pU/norm_pU

    pB_pU = pB - pU
    dot_pU = np.dot(pU, pU)
    dot_pB_pU = np.dot(pB_pU, pB_pU)
    dot_pU_pB_pU = np.dot(pU, pB_pU)
    fact = dot_pU_pB_pU ** 2 - dot_pB_pU * (dot_pU - trust_radius ** 2)
    tau = (-dot_pU_pB_pU + np.sqrt(fact)) / dot_pB_pU

    return pU + tau * pB_pU

def trust_region_dogleg(func, x0, jacb=None, hess=None,  initial_trust_radius=1.0,
                        max_trust_radius=1000.0, eta=0.15, gtol=14-4, maxiter=None):

    '''
    :param func: function
    :param jacb: jacbian function
    :param hess: hessian matrix
    :param x0:  initial point
    :param initial_trust_radius:
    :param max_trust_radius:
    :param eta:
    :param gtol: terminal value
    :param maxiter:
    :return:
            f(xk) - f(xk+pk)
    rho = -------------------
            mk(0) - mk(pk)

    mk(p) = fk + gkT p + 0.5 * pT*Bk*P
    '''



    if jacb is None:
        raise ValueError('Jacobian is currently required for trust-region methods')
    if hess is None:
        raise ValueError('Hessian is currently required for trust-region methods')
    if not (0 <= eta < 0.25):
        raise Exception('invalid acceptance stringency')
    if max_trust_radius <= 0:
        raise Exception('the max trust radius must be positive')
    if initial_trust_radius <= 0:
        raise ValueError('the initial trust radius must be positive')
    if initial_trust_radius >= max_trust_radius:
        raise ValueError('the initial trust radius must be less than the '
                         'max trust radius')


    if maxiter is None:
        maxiter = len(x0)*200

    xk=x0
    trust_radius = initial_trust_radius
    k = 0
    while True:

        gk = jacb(xk)
        Bk = hess(xk)
        pk = dogleg(gk, Bk, trust_radius)

        act_red = func(xk) - func(xk+pk)
        pred_red = -(np.dot(gk, pk) + 0.5 * np.dot(pk, np.dot(Bk, pk)))

        if(pred_red==0):
            rhok = 1e99
        else:
            rhok = act_red/pred_red

        norm_pk = np.sqrt(pk.T.dot(pk))

        if(rhok<0.25):
            trust_radius = 0.25 * trust_radius
        else:
            if rhok > 0.75 and norm_pk == trust_radius:
                trust_radius = min(2.0 * trust_radius, max_trust_radius)
            else:
                trust_radius = trust_radius

        # Choose the position for the next iteration.
        if rhok > eta:
            xk = xk + pk
        else:
            xk = xk

        # Check if the gradient is small enough to stop
        if np.linalg.norm(gk) < gtol:
            break

        # Check if we have looked at enough iterations
        if k >= maxiter:
            break
        k = k + 1
    return xk



if __name__ == '__main__':

    import scipy.optimize

    res = scipy.optimize.minimize(func, (5,5), method='dogleg', jac=jacb, hess=hess)
    print(res.x)
    result = trust_region_dogleg(func, [5, 5], jacb, hess)
    print(result)

