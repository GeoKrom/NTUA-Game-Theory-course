import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve_continuous_are

# =========================================================
# 🔷 System parameters
# =========================================================
N = 5
dim = 2
T = 10
dt = 0.001
steps = int(T/dt)
sigma = 0.1  # noise

# Linear system
A = np.array([[0, 1],
              [-1, -1]])

B = np.array([[0],
              [1]])

# LQR weights
Q = np.eye(dim)
R = np.array([[10]])

# =========================================================
# 🔷 Solve Riccati Equation
# =========================================================
P = solve_continuous_are(A, B, Q, R)

K = np.linalg.inv(R) @ B.T @ P

print("Riccati P:\n", P)
print("Feedback gain K:\n", K)

# =========================================================
# 🔷 Multi-agent setup
# =========================================================
x = np.random.randn(N, dim)

trajectory = np.zeros((steps, N, dim))
u_traj = np.zeros((steps, N, 1))

# Fully connected Laplacian
L = N * np.eye(N) - np.ones((N, N))

# =========================================================
# 🔷 Simulation
# =========================================================
for k in range(steps):
    x_dot = np.zeros_like(x)
    
    for i in range(N):
        # consensus term
        consensus = np.zeros(dim)
        for j in range(N):
            if j != i:
                consensus += (x[i] - x[j])
        
        # LQR-based control
        u_i = -K @ consensus   # shape (1,)
        
        # noise
        w = sigma * np.random.randn(dim)
        
        # dynamics
        x_dot[i] = A @ x[i] + B.flatten() * u_i + w
        
        u_traj[k, i, 0] = u_i
    
    trajectory[k] = x
    x = x + dt * x_dot

time = np.linspace(0, T, steps)

# =========================================================
# 🔷 FIGURE 1: Trajectories
# =========================================================
plt.figure()
for i in range(N):
    plt.plot(trajectory[:, i, 0], trajectory[:, i, 1])
plt.title("LQR-based Consensus Nash Trajectories")
plt.xlabel("x")
plt.ylabel("y")
plt.grid()

# =========================================================
# 🔷 FIGURE 2: States
# =========================================================
plt.figure()
for i in range(N):
    plt.plot(time, trajectory[:, i, 0])
plt.title("x_i(t)")
plt.grid()

plt.figure()
for i in range(N):
    plt.plot(time, trajectory[:, i, 1])
plt.title("y_i(t)")
plt.grid()

# =========================================================
# 🔷 FIGURE 3: Inputs
# =========================================================
plt.figure()
for i in range(N):
    plt.plot(time, u_traj[:, i, 0])
plt.title("Control inputs u_i(t)")
plt.xlabel("Time")
plt.ylabel("u")
plt.grid()

plt.show()