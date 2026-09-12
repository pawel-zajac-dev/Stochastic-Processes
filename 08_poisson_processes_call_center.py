import numpy as np
from scipy.stats import poisson

# --- 0. TRAINING (ESTIMATION) NOTE ---
# In this specific example, there is NO machine learning or training loop
# because we explicitly assumed a known, fixed parameter (lambda = 100). 
# If we were to "train" this model on real historical data, training a Poisson 
# process simply means finding the Maximum Likelihood Estimation (MLE):
# lambda_estimated = total_observed_events / total_observation_time.
# There are no hidden weights to adjust via gradient descent here.
# -------------------------------------

# --- 1. THEORY: Calculating Probabilities ---
lambda_1h = 100           # Intensity: 100 calls per hour (pre-defined, not trained)
t = 0.5                   # Observation window: 0.5 hours (30 minutes)
lambda_t = lambda_1h * t  # Expected events in this window (50)

# Probability of exactly 50 calls (Probability Mass Function)
prob_exactly_50 = poisson.pmf(50, lambda_t)

# Probability of more than 50 calls (1 - Cumulative Distribution Function)
prob_more_than_50 = 1 - poisson.cdf(50, lambda_t)

print("--- THEORETICAL ANALYSIS ---")
print(f"Expected calls in 30 min: {lambda_t}")
print(f"Probability of exactly 50 calls: {prob_exactly_50:.4f}")
print(f"Probability of >50 calls: {prob_more_than_50:.4f}")

# --- 2. PRACTICE: Simulating Incoming Calls ---
# Generate continuous inter-arrival times using the exponential distribution.
# The 'scale' must be the inverse of our assumed intensity (1/100).
inter_arrival_times = np.random.exponential(scale=1/lambda_1h, size=5)

# Convert isolated gaps into absolute timestamps via cumulative sum
arrival_times = np.cumsum(inter_arrival_times)

print("\n--- SIMULATION: FIRST 5 CALLS ---")
for i, time_hours in enumerate(arrival_times, 1):
    minutes = int(time_hours * 60)
    seconds = int((time_hours * 3600) % 60)
    print(f"Call {i}: at {minutes} min and {seconds} sec from start")