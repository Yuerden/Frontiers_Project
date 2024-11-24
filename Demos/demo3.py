import numpy as np
import matplotlib.pyplot as plt

# Network Parameters
N = 5  # Number of nodes

# Adjacency matrix representing the network topology
# Visual representation:
# C1 <---> S1 <---> G <---> S2 <---> C2
A = np.array([
    [0, 1, 1, 0, 0],  # Node 0 connections (Generator)
    [1, 0, 0, 1, 0],  # Node 1 connections (Substation1)
    [1, 0, 0, 0, 1],  # Node 2 connections (Substation2)
    [0, 1, 0, 0, 0],  # Node 3 connections (Consumer1)
    [0, 0, 1, 0, 0],  # Node 4 connections (Consumer2)
])

# Initial State Variables
x_initial = np.array([100.0, 50.0, 50.0, 20.0, 20.0])

# Desired Power Levels for Generators and Substations
x_desired = np.array([100.0, 80.0, 80.0, 0.0, 0.0])

# Regulation Rates (for Generators and Substations) and Consumption Rates (for Consumers)
r = np.array([0.1, 0.05, 0.05, 0.0, 0.0])  # Regulation rates
d = np.array([0.0, 0.0, 0.0, 0.1, 0.1])    # Consumption rates

# Simulation Parameters
iterations = 200  # Total number of iterations
perturbation_iteration = 100  # When to apply the perturbation
perturbation_node = 0        # Node to perturb (e.g., Generator)
c = 80.0                     # Perturbation magnitude (reduce x_i by c)

# Beta values to test 
# DEMO HHHHHHHHHHEEEEEEEEEEEEEERRRRRRRRRRRRRRRRRRREEEEEEEEEEEEEEEEEE TTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTT
beta_values_to_test = [0.005, 0.02, 0.05, 0.08, 0.1]
# beta_values_to_test = [0.05, 0.2, 0.5, 0.53, 0.54] # TTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTT

# Colors for plotting different beta simulations
beta_colors = ['blue', 'green', 'red', 'purple', 'orange']

# Lists to store results for each beta value
results = {}

for idx, Beta in enumerate(beta_values_to_test):
    # Reset initial conditions for each simulation
    x = x_initial.copy()
    Beta_initial = Beta  # Store initial Beta value
    beta_values = [Beta]  # List to store Beta values over time
    x_values = [x.copy()]  # State variables over time
    x_eff_list = [np.mean(x)]  # Effective state variable over time
    beta_eff_list = [Beta * (np.sum(A) / N)]  # Effective coupling strength over time
    time_steps = [0]  # Time steps for plotting
    perturbations = []  # List to record when perturbations occur

    # Simulation Loop
    for t in range(1, iterations + 1):
        x_current = x_values[-1].copy()
        x_new = x_current.copy()
        
        # Compute the change for each node
        for i in range(N):
            # Intrinsic dynamics
            if r[i] > 0:
                # Generators and Substations try to reach desired power levels
                F = r[i] * (x_desired[i] - x_current[i])
            elif d[i] > 0:
                # Consumers consume power
                F = -d[i] * x_current[i]
            else:
                F = 0.0  # No intrinsic dynamics for nodes without regulation or consumption
            
            # Interaction with connected nodes
            G_sum = 0.0
            for j in range(N):
                if A[i][j] == 1:
                    G_sum += Beta * (x_current[j] - x_current[i])
            
            # Total change for node i
            dx_dt = F + G_sum
            x_new[i] += dx_dt  # Update the state variable
        
        # Apply perturbation at the specified iteration
        if t == perturbation_iteration:
            x_new[perturbation_node] -= c  # Reduce x_i by c
            perturbations.append({'iteration': t, 'type': f'Perturbation on Node {perturbation_node}', 'value': -c})
            print(f"Beta {Beta_initial}: Iteration {t}: Applied perturbation of {-c} to Node {perturbation_node}.")

        # Update the state variables for the next iteration
        x_values.append(x_new.copy())
        
        # Compute effective state variable x_eff
        x_eff = np.mean(x_new)
        x_eff_list.append(x_eff)
        
        # Compute effective coupling strength beta_eff
        avg_degree = np.sum(A) / N  # Compute average degree
        beta_eff = Beta * avg_degree
        beta_eff_list.append(beta_eff)
        
        # Store the current Beta value
        beta_values.append(Beta)
        
        # Record the time step
        time_steps.append(t)
        
        # Update x for the next iteration
        x = x_new.copy()
    
    # Store results for this Beta value
    results[Beta_initial] = {
        'time_steps': time_steps,
        'x_values': np.array(x_values).T,
        'x_eff_list': x_eff_list,
        'beta_values': beta_values,
        'beta_eff_list': beta_eff_list,
        'perturbations': perturbations,
        'color': beta_colors[idx % len(beta_colors)]
    }

# After all simulations, plot the Effective State Variable Over Time for each Beta

plt.figure(figsize=(12, 8))
for Beta in beta_values_to_test:
    res = results[Beta]
    plt.plot(res['time_steps'], res['x_eff_list'],
             label=f'$\\beta$ = {Beta}',
             color=res['color'],
             linewidth=2)
    # Mark perturbations
    for p in res['perturbations']:
        plt.axvline(x=p['iteration'], color=res['color'], linestyle='--', linewidth=1)
        plt.text(p['iteration'], min(res['x_eff_list']), f"Perturbation", rotation=90,
                 verticalalignment='bottom', fontsize=8, color=res['color'])

plt.xlabel('Iteration')
plt.ylabel('Effective State $x_{eff}$')
plt.title('Effective State Variable Over Time for Different $\\beta$ Values')
plt.ylim(10, 60)
plt.legend()
plt.grid(True)
plt.show()
