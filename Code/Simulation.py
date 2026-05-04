import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve_continuous_are

# Parameters
N = 5
dim = 2
T = 10
dt = 0.01
steps = int(T/dt)
sigma = 0.1

# System
A = np.array([[0, 1],
              [-1, -1]])

B = np.array([[0],
              [1]])

Q = np.eye(dim)
R = np.array([[1]])

# Riccati
P = solve_continuous_are(A, B, Q, R)
K = np.linalg.inv(R) @ B.T @ P

# Initial state
x = np.random.randn(N, dim)
total_effort = 0
trajectory = np.zeros((steps, N, dim))
u_traj = np.zeros((steps, N))

# Simulation
for k in range(steps):
    x_dot = np.zeros_like(x)

    for i in range(N):
        consensus = np.zeros(dim)
        for j in range(N):
            if j != i:
                consensus += (x[i] - x[j])

        u_i = -K @ consensus
        w = sigma * np.random.randn(dim)
        total_effort += np.linalg.norm(u_i)**2 * dt
        x_dot[i] = A @ x[i] + B.flatten() * u_i + w
        u_traj[k, i] = u_i

    trajectory[k] = x
    x = x + dt * x_dot

time = np.linspace(0, T, steps)
print(total_effort)
# =========================================================
# 🔷 1. Trajectories
# =========================================================
plt.figure()
for i in range(N):
    plt.plot(trajectory[:, i, 0], trajectory[:, i, 1],
             label=f"Agent {i+1}")
plt.title("Agent Trajectories")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid()

# =========================================================
# 🔷 2. x_i(t)
# =========================================================
plt.figure()
for i in range(N):
    plt.plot(time, trajectory[:, i, 0],
             label=f"$x_{i+1}(t)$")
plt.title("State Components $x_{i1}(t)$")
plt.xlabel("Time (s)")
plt.ylabel("x")
plt.legend()
plt.grid()

# =========================================================
# 🔷 3. x_i2(t)
# =========================================================
plt.figure()
for i in range(N):
    plt.plot(time, trajectory[:, i, 1],
             label=f"$y_{i+1}(t)$")
plt.title("State Components $x_{i2}(t)$")
plt.xlabel("Time (s)")
plt.ylabel("y")
plt.legend()
plt.grid()

# =========================================================
# 🔷 4. u_i(t)
# =========================================================
plt.figure()
for i in range(N):
    plt.plot(time, u_traj[:, i],
             label=f"$u_{i+1}(t)$")
plt.title("Control Inputs $u_i(t)$")
plt.xlabel("Time (s)")
plt.ylabel("u")
plt.legend()
plt.grid()

plt.show()