

import numpy as np

def jacb(func):
    return

def hess(func):
    return


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

    pU = - (gk.T.dot(gk)).dot(np.linalg.inv(gk.T.dot(Bk).dot(gk))).dot(gk)

    norm_pU = np.sqrt(np.dot(pU, pU))

    # caucy point is outside of the trust-region
    if(norm_pU > trust_radius):
        return trust_radius * pU/norm_pU

    pB_pU = pB - pU
    dot_pB_pU = np.dot(pB_pU, pB_pU)
    dot_pU_pB_pU = np.dot(pU, pB_pU)
    fact = dot_pU_pB_pU ** 2 - dot_pB_pU * (dot_pU - trust_radius ** 2)
    tau = (-dot_pU_pB_pU + np.sqrt(fact)) / dot_pB_pU

    return pU + tau(pB_pU)

def trust_region_dogleg(func, jacb, hess, x0, initial_trust_radius=1.0,
                        max_trust_radius=100.0, eta=0.15, gt0l=14-4, maxiter=100):

    xk=x0
    trust_radius = initial_trust_radius
    k = 0
    while True:

        gk = jacb(xk)
        Bk = hess(xk)
        pk = dogleg(gk, Bk, trust_radius)

        act_red = func(xk) - func(xk+pk)
        pred_red = -(np.dot(gk, pk) + 0.5 * np.dot(pk, np.dot(Bk,pk)))

        if(pred_red==0):
            rhok = 1e99
        else:
            rhok = act_red/pred_red

        norm_pk = np.sqrt(pk.T.dot(pk))

        if(rhok<0.25):
            trust_radius = 9.25* norm_pk
        else:
            if rhok > 0.75 and norm_pk == trust_radius:
                trust_radius = min(2.0 * trust_radius, max_trust_radius)
            else:
                trust_radius = trust_radius

