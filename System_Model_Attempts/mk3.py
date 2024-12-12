import numpy as np

# Number of generators and loads
g = 2  # Number of generators
l = 8  # Number of loads
N = g + l  # Total number of nodes

# Generator capacities (constants)
P_gen_max = np.ones(g) * 100  # Each generator has a capacity of 100 units

# Load desired power (constants)
P_load_desired = np.ones(l) * 50  # Each load desires 50 units of power

# Initialize producing power
P_gen = np.zeros(g)  # Generators start at full capacity

# State variables
x = P_gen / P_gen_max  # Generators' capacity utilization

# Initialize adjacency matrix with zeros
A = np.zeros((N, N))

# Connect each generator to all loads (fully connected bipartite graph)
# for i in range(1):  # Generators
for i in range(g):  # Generators
    for j in range(g, N):  # Loads
        A[i][j] = 1  # Connection from generator i to load j
        A[j][i] = 1  # Connection from load j to generator i

alpha_gen = 0.01  # Decay rate for generators
beta = 0.1  # Interaction strength

num_iterations = 20

# Function to find connected components
def find_connected_components(A, N):
    visited = [False] * N
    components = []
    for v in range(N):
        if not visited[v]:
            component = []
            queue = [v]
            visited[v] = True
            while queue:
                node = queue.pop(0)
                component.append(node)
                neighbors = [i for i in range(N) if A[node][i] == 1 and not visited[i]]
                for neighbor in neighbors:
                    visited[neighbor] = True
                    queue.append(neighbor)
            components.append(component)
    return components

for t in range(num_iterations):
    # Remove all links at iteration 20
    if t == 20:
        A = np.zeros((N, N))

    dx = np.zeros(g)

    # Find connected components
    components = find_connected_components(A, N)

    # Initialize loads' actual power received and state variables
    P_load = np.zeros(l)
    y = np.zeros(l)

    # For each connected component
    for comp in components:
        # Separate generators and loads in the component
        comp_gens = [i for i in comp if i < g]
        comp_loads = [i - g for i in comp if i >= g]

        if comp_gens and comp_loads:
            # Calculate total supply and demand within the component
            total_supply = np.sum(P_gen[comp_gens])
            total_demand = np.sum(P_load_desired[comp_loads])

            # Determine allocation factor
            if total_supply >= total_demand:
                eta = 1.0
            else:
                eta = total_supply / total_demand

            # Update loads' actual power received and state variables
            P_load[comp_loads] = eta * P_load_desired[comp_loads]
            y[comp_loads] = P_load[comp_loads] / P_load_desired[comp_loads]
        else:
            # If there are no generators or loads in the component
            # Loads receive no power
            P_load[comp_loads] = 0
            y[comp_loads] = 0

    # Update generators
    for i in range(g):
        F_i = -alpha_gen * x[i]
        # Unmet demand of loads connected to generator i
        connected_loads = [j - g for j in range(g, N) if A[i][j] == 1]
        if connected_loads:
            unmet_demand = np.sum(P_load_desired[connected_loads] - P_load[connected_loads])
            G_i = beta * (1 - x[i]) * (unmet_demand / P_gen_max[i])
        else:
            G_i = 0
        dx_i_dt = F_i + G_i
        dx[i] = dx_i_dt

    # Update generators' state variables
    x += dx
    x = np.clip(x, 0, 1)
    P_gen = x * P_gen_max

    # Optionally, print or store the results at each iteration
    print(f"Iteration {t+1}:")
    print(f"Generator states x: {x}")
    print(f"Load states y: {y}\n")
