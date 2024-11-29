import numpy as np
import ruptures as rpt
import matplotlib.pyplot as plt
from scipy.stats import bernoulli

# Function to generate Bernoulli samples with one change point
def generate_bernoulli_sequence(T, tau, p1, p2):
    seq = np.concatenate([
        bernoulli.rvs(p1, size=tau),
        bernoulli.rvs(p2, size=T-tau)
    ])
    return seq

# Function to calculate theoretical Pd based on the formula
def calculate_pd(T, p1, p2):
    KL_divergence = p1 * np.log(p1 / p2) + (1 - p1) * np.log((1 - p1) / (1 - p2))
    delta = np.abs(p1 - p2)
    Pd = 1 - np.exp(-T * KL_divergence) - np.exp(-T * delta ** 2)
    return Pd

# Parameters
T = 100  # Length of the sequence
num_cases = 100  # Number of different cases
p1, p2 = 0.2, 0.8  # Parameters for the Bernoulli distributions

# To store results
accuracies = []
Pd_values = []

# Loop over 100 different cases
for case in range(num_cases):
    # Randomly choose a change point location
    tau = np.random.randint(20, T-20)  # Ensure the change point is not too close to edges
    
    # Generate the Bernoulli sequence
    sequence = generate_bernoulli_sequence(T, tau, p1, p2)
    
    # Apply the PELT algorithm with L2 cost function
    algo = rpt.Pelt(model="l2").fit(sequence)
    result = algo.predict(pen=1)  # You can adjust the penalty value for better results
    
    # Detected change point (PELT returns the end of the last segment, so subtract 1)
    detected_change_point = result[0] - 1
    
    # Calculate accuracy: 1 if correctly detected, 0 otherwise
    accuracy = 1 if detected_change_point == tau else 0
    accuracies.append(accuracy)
    
    # Calculate the theoretical Pd value
    Pd = calculate_pd(T, p1, p2)
    Pd_values.append(Pd)

# Plot comparison between accuracy and theoretical Pd
plt.figure(figsize=(10, 6))
plt.plot(Pd_values, label='Theoretical $P_d$', marker='o', linestyle='--', color='blue')
plt.plot(accuracies, label='Detection Accuracy', marker='x', linestyle='-', color='red')
plt.xlabel('Case Number')
plt.ylabel('Value')
plt.title('Comparison of Theoretical $P_d$ vs. Detection Accuracy')
plt.legend()
plt.show()
