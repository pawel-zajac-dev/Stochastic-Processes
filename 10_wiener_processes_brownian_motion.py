import numpy as np

# 1. Simulation parameters
T = 1.0          # Total observation time (e.g., 1 second)
N = 10           # Number of time steps (reduced to 10 to easily track values)
dt = T / N       # Length of a single time step (here 0.1 s)

# Time axis from 0 to 1 with equal intervals (for N=10 we need 11 points to include zero)
time_steps = np.linspace(0.0, T, N + 1)

# 2. Generating independent increments (Gaussian white noise)
# CRITICAL POINT: The scale (standard deviation) must be the square root of dt
dW = np.random.normal(loc=0.0, scale=np.sqrt(dt), size=N)

# 3. Assembling the full trajectory
# We enforce the W_0 = 0 axiom and append the cumulative sum of increments
W = np.concatenate(([0.0], np.cumsum(dW)))

# 4. Displaying the results (step-by-step simulation)
print("--- WIENER PROCESS SIMULATION ---")
for i in range(N + 1):
    print(f"Step {i:2} | Time: {time_steps[i]:.1f}s | Position W_t: {W[i]:.4f}")