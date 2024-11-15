import numpy as np
import matplotlib.pyplot as plt

# Network Parameters
N = 5  # Number of nodes
A = np.array([[0, 1, 1, 0, 0],
              [1, 0, 0, 1, 0],
              [1, 0, 0, 0, 1],
              [0, 1, 0, 0, 0],
              [0, 0, 1, 0, 0]])

# Initial State Variables
x = np.array([100.0, 50.0, 50.0, 20.0, 20.0])  # Power levels at nodes
x_desired = np.array([100.0, 80.0, 80.0, 0.0, 0.0])  # Target states
r = np.array([0.1, 0.05, 0.05, 0.0, 0.0])  # Regulation rates for generators and substations
d = np.array([0.0, 0.0, 0.0, 0.1, 0.1])    # Consumption rates for consumers

# Simulation Parameters
initial_Beta = 0.02  # Initial transmission line conductance
beta_increment = 0.1  # Amount to increase Beta each phase
iterations_per_beta = 18  # Number of iterations to run for each Beta level
max_beta = 0.8  # Maximum Beta value before halting

# Lists to store computed metrics over time
x_values = [x.copy()]
x_eff_list = [np.mean(x)]
f_values = []
beta_values = [initial_Beta]

# Simulation Loop with Gradual Beta Increase
iterations_cum = 0
Beta = initial_Beta
while Beta <= max_beta:
    for t in range(iterations_per_beta):
        x_current = x_values[-1].copy()
        x_new = x_current.copy()
        
        # Compute the change for each node
        for i in range(N):
            # Intrinsic dynamics
            if r[i] > 0:
                # Generators and Substations try to reach desired power levels
                f = r[i] * (x_desired[i] - x_current[i])
            elif d[i] > 0:
                # Consumers consume power
                f = -d[i] * x_current[i]
            else:
                f = 0.0  # No intrinsic dynamics for nodes without regulation or consumption
            
            # Interaction with connected nodes
            g_sum = 0.0
            for j in range(N):
                if A[i][j] == 1:
                    g_sum += Beta * (x_current[j] - x_current[i])
            
            # Total change for node i
            dx_dt = f + g_sum
            x_new[i] += dx_dt
        
        # iterations_cum+=1
        # if iterations_cum == 138:
        #     x_new[:] = 0

        # Store the new state
        x_values.append(x_new.copy())
        
        # Calculate effective state variable x_eff and resilience function f(Beta_eff, x_eff)
        x_eff = np.mean(x_new)
        x_eff_list.append(x_eff)
        
        # Compute F(x_eff) and G(x_eff, x_eff) for resilience function
        F_x_eff = -np.mean(r[r > 0]) * (x_eff - np.mean(x_desired))
        G_x_eff = Beta * x_eff * (1 - x_eff / np.mean(x_desired))
        f_values.append(F_x_eff + G_x_eff)
        
        # Store current Beta value
        beta_values.append(Beta)
    
    # Increase Beta for the next phase
    Beta += beta_increment

# Convert lists to numpy arrays for easier indexing
x_values_transposed = np.array(x_values).T
x_eff_array = np.array(x_eff_list)
f_array = np.array(f_values)
beta_array = np.array(beta_values[:-1])  # Match the length of f_values

# Plotting
fig, axs = plt.subplots(3, 1, figsize=(10, 18))

# Plot Node States Over Iterations
for i in range(N):
    axs[0].plot(range(len(x_values)), x_values_transposed[i], label=f'Node {i}')
axs[0].set_title("Node States Over Iterations with Increasing Beta")
axs[0].set_xlabel("Iteration")
axs[0].set_ylabel("State (x_i)")
axs[0].legend()
axs[0].set_ylim(-20, 150)  # Set y-axis limits to focus on the desired range
axs[0].grid(True)

# Plot Effective State Variable Over Iterations with Beta on Secondary Axis
ax2 = axs[1].twinx()  # Create a secondary y-axis
axs[1].plot(range(len(x_eff_array)), x_eff_array, label="Effective State (x_eff)", color='magenta')
ax2.plot(range(len(beta_array)), beta_array, label="Beta", color='blue', linestyle="--")
axs[1].set_title("Effective State Variable (x_eff) Over Iterations with Beta")
axs[1].set_xlabel("Iteration")
axs[1].set_ylabel("Effective State (x_eff)", color='magenta')
ax2.set_ylabel("Beta", color='blue')
axs[1].grid(True)

# Plot Resilience Function Over Iterations
axs[2].plot(range(len(f_array)), f_array, label="Resilience Function (f)", color='teal')
axs[2].set_title("Resilience Function (f) Over Iterations with Increasing Beta")
axs[2].set_xlabel("Iteration")
axs[2].set_ylabel("f(Beta_eff, x_eff)")
axs[2].grid(True)
axs[2].set_ylim(-40, 15)

# Optional: Mark critical points or phase transitions visually
critical_indices = [i for i, f_val in enumerate(f_array) if f_val <= 0]  # Indices where f reaches or drops below zero
for ax in axs:
    for idx in critical_indices:
        ax.axvline(x=idx, color='red', linestyle='--', linewidth=0.5, label="Tipping Point" if idx == critical_indices[0] else "")
    ax.legend()

plt.tight_layout()
plt.show()
