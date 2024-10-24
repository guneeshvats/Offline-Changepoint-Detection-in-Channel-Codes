import numpy as np
import ruptures as rpt
import matplotlib.pyplot as plt
from scipy.stats import bernoulli, norm

# Function to generate Bernoulli samples with one change point
def generate_bernoulli_sequence(T, tau, p1, p2):
    seq = np.concatenate([
        bernoulli.rvs(p1, size=tau),
        bernoulli.rvs(p2, size=T-tau)
    ])
    return seq

# Function to calculate sigma using Fisher Information with a lower bound epsilon
def calculate_sigma(T, tau, p1, p2, epsilon=1e-5):
    fisher_information = (tau / (p1 * (1 - p1))) + ((T - tau) / (p2 * (1 - p2)))
    # Prevent sigma from becoming too small by introducing a lower bound epsilon
    sigma = max(np.sqrt(1 / fisher_information), epsilon)
    return sigma

# Function to calculate Pd using the normal distribution and sigma
def calculate_pd(delta, sigma):
    # Pd based on the cumulative distribution function (CDF) of the normal distribution
    Pd = norm.cdf(delta / sigma) - norm.cdf(-delta / sigma)
    return Pd

# Parameters
T_values = range(100, 1001, 10)  # Sequence lengths from 100 to 1000
p1, p2 = 0.2, 0.8  # Parameters for the Bernoulli distributions
delta = 5  # Tolerance margin for detected change point
num_cases = 100  # Number of cases per T value
epsilon = 1e-5  # Small positive value to prevent sigma from becoming too small

# To store results for each T value
avg_Pd_values_T = []
avg_accuracies_T = []
T_list = []

# Loop over increasing values of T
for T in T_values:
    Pd_values = []
    accuracies = []
    
    # Run 100 cases for each T
    for case in range(num_cases):
        # Randomly choose a change point location
        tau = np.random.randint(10, T-10)  # Ensure the change point is not too close to edges
        
        # Generate the Bernoulli sequence
        sequence = generate_bernoulli_sequence(T, tau, p1, p2)
        
        # Apply the Binary Segmentation algorithm with L2 cost function
        algo = rpt.Binseg(model="l2").fit(sequence)
        result = algo.predict(n_bkps=1)  # We specify that there is exactly 1 change point
        
        # Detected change point (Binseg returns the end of the last segment, so subtract 1)
        detected_change_point = result[0] - 1
        
        # Calculate accuracy: 1 if within tolerance delta, 0 otherwise
        accuracy = 1 if abs(detected_change_point - tau) <= delta else 0
        accuracies.append(accuracy)
        
        # Compute sigma using Fisher Information, ensuring it doesn't become too small
        sigma = calculate_sigma(T, tau, p1, p2, epsilon)
        
        # Calculate the theoretical Pd value using the normal CDF formula
        Pd = calculate_pd(delta, sigma)
        Pd_values.append(Pd)
    
    # Calculate the average Pd and accuracy for this T
    avg_Pd = np.mean(Pd_values)
    avg_accuracy = np.mean(accuracies)
    
    avg_Pd_values_T.append(avg_Pd)
    avg_accuracies_T.append(avg_accuracy)
    T_list.append(T)
    
    # Print results for each T
    print(f"T = {T}: Average Accuracy = {avg_accuracy}, Average Pd = {avg_Pd}")

# Plot comparison of average accuracy and average Pd as T increases
plt.figure(figsize=(10, 6))
plt.plot(T_list, [pd * 100 for pd in avg_Pd_values_T], label='Average Theoretical $P_d$', marker='o', linestyle='--', color='blue')
plt.plot(T_list, [acc * 100 for acc in avg_accuracies_T], label='Average Detection Accuracy', marker='x', linestyle='-', color='red')
plt.xlabel('Sequence Length (T)')
plt.ylabel('Percentage (%)')  # Change y-axis label to percentage
plt.ylim(0, 110)  # Set y-axis from 0 to 100
plt.title(f'Comparison of Average $P_d$ vs. Average Detection Accuracy as T increases (delta={delta})')
plt.legend()
plt.show()
