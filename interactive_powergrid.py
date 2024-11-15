import numpy as np
import matplotlib.pyplot as plt

# Network Parameters
N = 5  # Number of nodes
# Adjacency matrix representing the network topology
A = np.array([
    [0, 1, 1, 0, 0],  # Node 0 connections (Generator)                               C1  <->  S1  <->  G   <->  S2  <-> C2
    [1, 0, 0, 1, 0],  # Node 1 connections (Substation)                                               / \
    [1, 0, 0, 0, 1],  # Node 2 connections (Substation)                                             S1   S2
    [0, 1, 0, 0, 0],  # Node 3 connections (Consumer)                                              /       \
    [0, 0, 1, 0, 0],  # Node 4 connections (Consumer)                                            C1         C2
])

# Initial State Variables
# Power levels at nodes: [Generator, Substation1, Substation2, Consumer1, Consumer2]
x = np.array([100.0, 50.0, 50.0, 20.0, 20.0])

# Desired Power Levels for Generators and Substations
x_desired = np.array([100.0, 80.0, 80.0, 0.0, 0.0])

# Regulation Rates (for Generators and Substations) and Consumption Rates (for Consumers)
r = np.array([0.1, 0.05, 0.05, 0.0, 0.0])  # Regulation rates
d = np.array([0.0, 0.0, 0.0, 0.1, 0.1])    # Consumption rates

# Coupling Constant (Initial Value)
Beta = 0.02  # Transmission line conductance

# Simulation Parameters
iterations = 500  # Total number of iterations

# Lists to store computed metrics over time
x_values = [x.copy()]          # State variables
beta_values = [Beta]           # Beta values
x_eff_list = [np.mean(x)]      # Effective state variable
f_values = []
beta_eff_list = [Beta * (np.sum(A) / N)]  # Effective coupling strength

# Average degree (used for beta_eff calculation)
avg_degree = np.sum(A) / N  # Total connections divided by number of nodes

# Simulation Loop
for t in range(iterations):
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
        f_array = np.array(f_values)
    
    # Introduce shocks or change Beta every 10 iterations
    if t % 4 == 0 and t > 0:
        # Plot the current state variables
        time_steps = range(len(x_values))
        x_values_transposed = np.array(x_values).T
        plt.figure(figsize=(12, 8))
        
        # Subplot 1: Power Levels Over Time
        plt.subplot(3, 1, 1)
        for i in range(N):
            plt.plot(time_steps, x_values_transposed[i], label=f'Node {i}')
        plt.xlabel('Iteration')
        plt.ylabel('Power Level x_i')
        plt.title(f'Power Levels Over Time (Up to Iteration {t})')
        plt.legend()
        plt.grid(True)
        
        # Subplot 2: Beta Values Over Time
        plt.subplot(2, 1, 2)
        plt.plot(time_steps, beta_values, label='Beta', color='red')
        plt.xlabel('Iteration')
        plt.ylabel('Beta')
        plt.title('Beta Values Over Time')
        plt.grid(True)
        plt.legend()

        plt.plot(range(len(f_array)), f_array, label="Resilience Function (f)", color='teal')
        plt.title("Resilience Function (f) Over Iterations with Increasing Beta")
        plt.xlabel("Iteration")
        plt.ylabel("f(Beta_eff, x_eff)")
        plt.grid(True)
        plt.ylim(-40, 15)
        
        plt.tight_layout()
        plt.show()
        
        # Introduce shocks or change Beta
        print(f"Iteration {t}:")
        change_input = input("Do you want to introduce a shock or change Beta? (yes/no): ").strip().lower()
        if change_input == 'yes' or change_input == 'y':
            # Option to introduce a shock to x_i
            print("Nodes: [Generator, Substation1, Substation2, Consumer1, Consumer2]")
            node_to_shock = int(input(f"Enter node number to shock (0 to {N-1}): "))
            shock_value = float(input("Enter shock value to add to x_i (e.g., -20 to decrease, 20 to increase): "))
            x_new[node_to_shock] += shock_value
            print(f"Applied shock of {shock_value} to node {node_to_shock}.")
            
            # Option to change Beta
            beta_input = input("Enter new Beta value (or press Enter to keep current Beta): ").strip()
            if beta_input != '':
                Beta = float(beta_input)
                print(f"Updated Beta to {Beta}.")
            else:
                print(f"Beta remains at {Beta}.")
        elif change_input == 'asdf':
            x_new[:] = 0  # Set all node states in `x` to 0, directly impacting subsequent iterations
            print("Set all node states to 0 due to 'asdf' input.")
        else:
            print("No changes made.")
    
    # Update the state variables for the next iteration
    x_values.append(x_new.copy())

    # Compute effective state variable x_eff
    x_eff = np.mean(x_new)
    x_eff_list.append(x_eff)
    
    # Compute effective coupling strength beta_eff
    beta_eff = Beta * avg_degree
    beta_eff_list.append(beta_eff)

    # Compute F(x_eff) and G(x_eff, x_eff) for resilience function
    F_x_eff = -np.mean(r[r > 0]) * (x_eff - np.mean(x_desired))
    G_x_eff = Beta * x_eff * (1 - x_eff / np.mean(x_desired))
    f_values.append(F_x_eff + G_x_eff)
    
    # Store the current Beta value
    beta_values.append(Beta)

# After Simulation: Plotting the Results
time_steps = range(len(x_values))
x_values_transposed = np.array(x_values).T

plt.figure(figsize=(12, 10))

# Subplot 1: Power Levels Over Time
plt.subplot(3, 1, 1)
line_styles = ['-', '--', '-.', ':', 'solid']
markers = ['o', 's', '^', 'D', 'x']
for i in range(N):
    plt.plot(time_steps, x_values_transposed[i],
             label=f'Node {i}',
             linestyle=line_styles[i % len(line_styles)],
             marker=markers[i % len(markers)],
             markevery=5)  # Plot markers every 5 data points
plt.xlabel('Iteration')
plt.ylabel('Power Level x_i')
plt.title('Power Levels Over Time')
plt.legend()
plt.grid(True)

# Subplot 2: Beta Values Over Time
plt.subplot(3, 1, 2)
plt.plot(time_steps, beta_values, label='Beta', color='red')
plt.xlabel('Iteration')
plt.ylabel('Beta')
plt.title('Beta Values Over Time')
plt.grid(True)
plt.legend()

# Subplot 3: Effective State Variable Over Time
plt.subplot(3, 1, 3)
plt.plot(time_steps, x_eff_list, label='x_eff', color='purple')
plt.xlabel('Iteration')
plt.ylabel('Effective State x_eff')
plt.title('Effective State Over Time')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()

# Plot x_eff vs beta_eff
plt.figure(figsize=(8, 6))
plt.scatter(beta_eff_list, x_eff_list, color='magenta')
plt.xlabel('Effective Coupling Strength beta_eff')
plt.ylabel('Effective State x_eff')
plt.title('x_eff vs beta_eff')
plt.grid(True)
plt.show()
