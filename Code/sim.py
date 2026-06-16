import numpy as np
import matplotlib.pyplot as plt

# ==========================================================
# PARAMETERS
# ==========================================================

N = 5

dt = 0.01
Tf = 30

steps = int(Tf/dt)

# ==========================================================
# AGENT DYNAMICS
#
# x_dot = A x + B u + w
#
# x = [position velocity]^T
# ==========================================================

A = np.array([
    [1,1],
    [0,0]
])

B = np.array([
    [0],
    [1]
])

# ==========================================================
# COMMUNICATION GRAPH
# Ring Topology
# ==========================================================

Adj = np.array([
    [0,1,0,0,1],
    [1,0,1,0,0],
    [0,1,0,1,0],
    [0,0,1,0,1],
    [1,0,0,1,0]
])

Deg = np.diag(np.sum(Adj,axis=1))

L = Deg - Adj

# ==========================================================
# IMITATION DYNAMICS
# ==========================================================

gamma = 0.5

# ==========================================================
# NOISE
# ==========================================================

sigma = 0.01

# ==========================================================
# INITIAL STATES
# ==========================================================

X = np.array([
    [-8, 2],
    [ 6,-2],
    [ 4, 1],
    [-3,-1],
    [10, 3]
],dtype=float)

# ==========================================================
# INITIAL STRATEGIES
# ==========================================================

theta = np.array([
    0.5,
    2.0,
    1.0,
    3.0,
    4.0
])

# ==========================================================
# STORAGE
# ==========================================================

state_hist = np.zeros((steps,N,2))

theta_hist = np.zeros((steps,N))

u_hist = np.zeros((steps,N))

consensus_hist = np.zeros((steps,N))

cost_hist = np.zeros((steps,N))

# ==========================================================
# COST WEIGHTS
# ==========================================================

Q = np.diag([1,2])

R = 1.0

rho = 1.0

# ==========================================================
# MAIN LOOP
# ==========================================================

for k in range(steps):

    state_hist[k] = X
    theta_hist[k] = theta

    U = np.zeros(N)

    # ------------------------------------------------------
    # CONTROL COMPUTATION
    # ------------------------------------------------------

    for i in range(N):

        ei = np.zeros(2)

        for j in range(N):

            if Adj[i,j] == 1:

                ei += X[i] - X[j]

        consensus_hist[k,i] = np.linalg.norm(ei)

        ui = -theta[i] * ei[0]

        U[i] = ui

        u_hist[k,i] = ui

        strategy_cost = 0

        for j in range(N):

            if Adj[i,j] == 1:

                strategy_cost += (
                    theta[i]-theta[j]
                )**2

        Ji = (
            ei.T @ Q @ ei
            +
            R*ui**2
            +
            rho*strategy_cost
        )

        cost_hist[k,i] = Ji

    # ------------------------------------------------------
    # STATE UPDATE
    # ------------------------------------------------------

    for i in range(N):

        noise = sigma*np.random.randn(2)

        xdot = (
            A @ X[i]
            +
            B.flatten()*U[i]
            +
            noise
        )

        X[i] += dt*xdot

    # ------------------------------------------------------
    # IMITATION UPDATE
    # ------------------------------------------------------

    theta_dot = np.zeros(N)

    for i in range(N):

        for j in range(N):

            if Adj[i,j] == 1:

                theta_dot[i] += (
                    theta[j]
                    -
                    theta[i]
                )

    theta += dt*gamma*theta_dot

# ==========================================================
# TIME VECTOR
# ==========================================================

t = np.arange(steps)*dt

# ==========================================================
# POSITION STATES
# ==========================================================

plt.figure(figsize=(8,5))

for i in range(N):

    plt.plot(
        t,
        state_hist[:,i,0],
        label=f'Agent {i+1}'
    )

plt.title('Position States')
plt.xlabel('Time [s]')
plt.ylabel('Position')
plt.grid(True)
plt.legend()
plt.show()

# ==========================================================
# VELOCITY STATES
# ==========================================================

plt.figure(figsize=(8,5))

for i in range(N):

    plt.plot(
        t,
        state_hist[:,i,1],
        label=f'Agent {i+1}'
    )

plt.title('Velocity States')
plt.xlabel('Time [s]')
plt.ylabel('Velocity')
plt.grid(True)
plt.legend()
plt.show()

# ==========================================================
# STRATEGY EVOLUTION
# ==========================================================

plt.figure(figsize=(8,5))

for i in range(N):

    plt.plot(
        t,
        theta_hist[:,i],
        label=f'θ{i+1}'
    )

plt.title('Imitation Dynamics')
plt.xlabel('Time [s]')
plt.ylabel('Strategy Parameter')
plt.grid(True)
plt.legend()
plt.show()

# ==========================================================
# CONTROL INPUTS
# ==========================================================

plt.figure(figsize=(8,5))

for i in range(N):

    plt.plot(
        t,
        u_hist[:,i],
        label=f'Agent {i+1}'
    )

plt.title('Control Inputs')
plt.xlabel('Time [s]')
plt.ylabel('u_i')
plt.grid(True)
plt.legend()
plt.show()

# ==========================================================
# CONTROL EFFORT
# ==========================================================

plt.figure(figsize=(8,5))

for i in range(N):

    plt.plot(
        t,
        u_hist[:,i]**2,
        label=f'Agent {i+1}'
    )

plt.title('Control Effort')
plt.xlabel('Time [s]')
plt.ylabel('u_i²')
plt.grid(True)
plt.legend()
plt.show()

# ==========================================================
# CONSENSUS ERRORS
# ==========================================================

plt.figure(figsize=(8,5))

for i in range(N):

    plt.plot(
        t,
        consensus_hist[:,i],
        label=f'Agent {i+1}'
    )

plt.title('Consensus Errors')
plt.xlabel('Time [s]')
plt.ylabel('||e_i||')
plt.grid(True)
plt.legend()
plt.show()

# ==========================================================
# LOCAL COSTS
# ==========================================================

plt.figure(figsize=(8,5))

for i in range(N):

    plt.plot(
        t,
        cost_hist[:,i],
        label=f'Agent {i+1}'
    )

plt.title('Local Nash Costs')
plt.xlabel('Time [s]')
plt.ylabel('J_i')
plt.grid(True)
plt.legend()
plt.show()

# ==========================================================
# PHASE PORTRAITS
# ==========================================================

plt.figure(figsize=(8,6))

for i in range(N):

    plt.plot(
        state_hist[:,i,0],
        state_hist[:,i,1],
        label=f'Agent {i+1}'
    )

plt.title('Phase Portrait')
plt.xlabel('Position')
plt.ylabel('Velocity')
plt.grid(True)
plt.legend()
plt.show()