import numpy as np
import matplotlib.pyplot as plt

# Network Parameters
N = 5  # Number of nodes

# Adjacency matrix representing the network topology
A = np.array([
    [0, 1, 1, 0, 0],  # Generator connected to both substations
    [1, 0, 0, 1, 0],  # Substation1 connected to Generator and Consumer1
    [1, 0, 0, 0, 1],  # Substation2 connected to Generator and Consumer2
    [0, 1, 0, 0, 0],  # Consumer1 connected to Substation1
    [0, 0, 1, 0, 0],  # Consumer2 connected to Substation2
])

# Initial State Variables
x = np.array([100.0, 80.0, 80.0, 20.0, 20.0])

# Desired Power Levels
x_desired = np.array([100.0, 80.0, 80.0, 0.0, 0.0])

# Regulation and Consumption Rates
r = np.array([0.1, 0.05, 0.05, 0.0, 0.0])  # Generators and Substations
d = np.array([0.0, 0.0, 0.0, 0.1, 0.1])    # Consumers

# Coupling Constant
Beta = 0.02

# Simulation Parameters
iterations = 100
perturbation_iteration = 50
perturbation_node = 0  # Perturb Generator
perturbation_value = -20.0  # Minor perturbation

# Lists to store variables
x_values = [x.copy()]
x_eff_list = [np.mean(x)]
variance_x_list = [np.var(x)]
f_eff_list = []

# Compute beta_eff
avg_degree = np.sum(A) / N
beta_eff = Beta * avg_degree

# Reset initial conditions
x = np.array([100.0, 50.0, 50.0, 20.0, 20.0])
x_values = [x.copy()]
x_eff_list = [np.mean(x)]
variance_x_list = [np.var(x)]
f_eff_list = []

# Simulation Loop
for t in range(1, iterations + 1):
    x_current = x_values[-1].copy()
    x_new = x_current.copy()
    F_array = np.zeros(N)
    
    # Update each node
    for i in range(N):
        # Intrinsic dynamics
        if r[i] > 0:
            F = r[i] * (x_desired[i] - x_current[i])
        elif d[i] > 0:
            F = -d[i] * x_current[i]
        else:
            F = 0.0
        F_array[i] = F  # Store F(x_i)
        
        # Interaction dynamics
        G_sum = 0.0
        for j in range(N):
            if A[i, j]:
                G = x_current[j] - x_current[i]
                G_sum += Beta * G
                
        # Update state
        x_new[i] += F + G_sum
    
    # Store variables
    x_values.append(x_new.copy())
    x_eff = np.mean(x_new)
    x_eff_list.append(x_eff)
    
    variance_x = np.var(x_new)
    variance_x_list.append(variance_x)
    
    # Compute F(x_eff)
    F_eff = np.mean(F_array)
    
    # Compute G(x_eff, x_eff) = beta_eff * Var(x_i)
    G_eff = beta_eff * variance_x
    
    # Compute f(beta_eff, x_eff)
    f_eff = F_eff + G_eff
    f_eff_list.append(f_eff)
    
# Convert lists to arrays
x_values = np.array(x_values)
f_eff_list = np.array(f_eff_list)
x_eff_list = np.array(x_eff_list)
variance_x_list = np.array(variance_x_list)

# Plotting
time_steps = range(iterations + 1)

plt.figure(figsize=(14, 6))

# Subplot 1: Node States Over Iterations
plt.subplot(1, 2, 1)
for i in range(N):
    plt.plot(time_steps, x_values[:, i], label=f'Node {i}')
plt.xlabel('Iteration')
plt.ylabel('Power Level $x_i$')
plt.title('Node States Over Iterations (Steady State)')
plt.legend()
plt.grid(True)

# Subplot 2: Resilience Function f(β_eff, x_eff) Over Iterations
plt.subplot(1, 2, 2)
plt.plot(time_steps[1:], f_eff_list, label='$f(\\beta_{eff}, x_{eff})$')
plt.xlabel('Iteration')
plt.ylabel('Resilience Function $f(\\beta_{eff}, x_{eff})$')
plt.title('Resilience Function Over Iterations')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
