import numpy as np
import matplotlib.pyplot as plt

# Parameters
N = 5
dim = 2
T = 10
dt = 0.01
steps = int(T/dt)
alpha = 1.0

# Individual targets (different for each agent)
c = np.random.randn(N, dim) * 5

# Initial conditions
x = np.random.randn(N, dim)

trajectory = np.zeros((steps, N, dim))
u_traj = np.zeros((steps, N, dim))

def neighbors(i):
    return [j for j in range(N) if j != i]

# Simulation
for k in range(steps):
    x_dot = np.zeros_like(x)
    
    for i in range(N):
        interaction = sum(x[i] - x[j] for j in neighbors(i))
        x_dot[i] = -((x[i] - c[i]) + alpha * interaction)
    
    u = x_dot.copy()
    
    trajectory[k] = x
    u_traj[k] = u
    
    x = x + dt * x_dot

time = np.linspace(0, T, steps)

# =========================================================
# 🔷 FIGURE 1: Trajectories
# =========================================================
plt.figure()
for i in range(N):
    plt.plot(trajectory[:, i, 0], trajectory[:, i, 1])
    plt.scatter(c[i,0], c[i,1], marker='x')  # targets
plt.title("Agent Trajectories")
plt.xlabel("x")
plt.ylabel("y")
plt.grid()

# =========================================================
# 🔷 FIGURE 2: X components
# =========================================================
plt.figure()
for i in range(N):
    plt.plot(time, trajectory[:, i, 0])
plt.title("States $x_i(t)$")
plt.xlabel("Time (s)")
plt.ylabel("x")
plt.grid()

# =========================================================
# 🔷 FIGURE 3: Y components
# =========================================================
plt.figure()
for i in range(N):
    plt.plot(time, trajectory[:, i, 1])
plt.title("States $y_i(t)$")
plt.xlabel("Time (s)")
plt.ylabel("y")
plt.grid()

# =========================================================
# 🔷 FIGURE 4: Control norm
# =========================================================
plt.figure()
for i in range(N):
    norm_u = np.linalg.norm(u_traj[:, i, :], axis=1)
    plt.plot(time, norm_u)
plt.title("Control Inputs $||u_i(t)||$")
plt.xlabel("Time (s)")
plt.ylabel("||u||")
plt.grid()

plt.show()