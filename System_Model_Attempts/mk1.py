import numpy as np

# Number of generators and loads
g = 2  # Number of generators
l = 8  # Number of loads
N = g + l  # Total number of nodes

# Generator capacities (constants)
P_gen_max = np.ones(g) * 100  # Example: each generator has a capacity of 100 units

# Load desired power (constants)
P_load_desired = np.ones(l) * 25  # Example: each load desires 10 units of power

# Initialize producing power and actual power
P_gen = np.zeros(g)   # Generators start at full capacity
P_load = np.zeros(l)  # Loads start receiving desired power

# State variables
x = P_gen / P_gen_max  # Should be an array of ones
y = P_load / P_load_desired  # Should be an array of ones

# Initialize adjacency matrix with zeros
A = np.zeros((N, N))

# Connect each generator to all loads (fully connected bipartite graph)
for i in range(g):  # Generators
    for j in range(g, N):  # Loads
        A[i][j] = 1  # Connection from generator i to load j
        A[j][i] = 1  # Connection from load j to generator i

alpha_gen = 0.01  # Decay rate for generators
alpha_load = 0.01  # Decay rate for loads
beta = 0.1  # Interaction strength

def F_gen(x_i, alpha_gen):
    return -alpha_gen * x_i

def F_load(y_i, alpha_load):
    return -alpha_load * y_i

def G_gen(P_load_desired_j, P_load_j, beta):
    return beta * (P_load_desired_j - P_load_j)

def G_load(P_load_desired_i, P_load_i, beta):
    return beta * (P_load_desired_i - P_load_i)

num_iterations = 50

for t in range(num_iterations):
    # Arrays to store the changes
    dx = np.zeros(g)
    dy = np.zeros(l)

    # Update generators
    for i in range(g):
        F_i = F_gen(x[i], alpha_gen)
        sum_G = 0
        for j in range(g, N):
            if A[i][j] == 1:
                # Load index adjusted for y array (starts at 0)
                y_j = y[j - g]
                sum_G += G_gen(P_load_desired[j - g], P_load[j - g], beta) / P_gen_max[i]
        dx_i_dt = F_i + sum_G
        dx[i] = dx_i_dt

    # Update loads
    for i in range(l):
        F_i = F_load(y[i], alpha_load)
        sum_G = 0
        for j in range(g):
            if A[i + g][j] == 1:
                x_j = x[j]
                sum_G += G_load(P_load_desired[i], P_load[i], beta) / P_load_desired[i]
        dy_i_dt = F_i + sum_G
        dy[i] = dy_i_dt

    # Update state variables
    x += dx  # For generators
    y += dy  # For loads

    # Ensure x and y remain within [0, 1]
    x = np.clip(x, 0, 1)
    y = np.clip(y, 0, 1)

    # Update producing power and actual power
    P_gen = x * P_gen_max
    P_load = y * P_load_desired

    # Optionally, print or store the results at each iteration
    print(f"Iteration {t+1}:")
    print(f"Generator states x: {x}")
    print(f"Load states y: {y}\n")
