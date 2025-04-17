from scipy.sparse.linalg import spsolve
import numpy as np

def newmark(q0, v0, M, K, dt, nt, gamma=0.5, beta=0.25):
    assert beta >= 0 and gamma >= 0.5, "The Newmark parameters are invalid (unstable simulation)"

    q_solution = np.zeros((len(q0), nt+1))
    q_solution[:, 0] = q0

    v_solution = np.zeros((len(v0), nt+1))
    v_solution[:, 0] = v0

    q_old = q0
    v_old = v0
    a_old = spsolve(M, - K @ q_old)

    A_newmark = (M + beta * dt**2 * K)

    for n in range(nt):
        b = - K @ (q_old + dt * v_old + (0.5 - beta) * dt**2 * a_old)
        a_new = spsolve(A_newmark, b)

        v_new = v_old + dt * ((1 - gamma) * a_old + gamma * a_new)
        q_new = q_old + dt * v_old + 0.5*dt**2*((1 - 2*beta)*a_old + 2*beta*a_new)

        q_solution[:, n+1] = q_new
        v_solution[:, n+1] = v_new

        q_old = q_new
        v_old = v_new
        a_old = a_new

    return q_solution, v_solution