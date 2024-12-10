import numpy as np

# Number of nodes (generators + loads)
num_nodes = 6

# Example: Node 0 and 1 are generators, Node 2-5 are loads
generators = [0, 1]
loads = [2, 3, 4, 5]

# Adjacency matrix (connection capacities)
A = np.array([
    [0, 0, 35, 0, 0, 0],  # Node 0 (Generator)
    [0, 0, 0, 15, 0, 0],  # Node 1 (Generator)
    [35, 0, 0, 5, 0, 0],  # Node 2 (Load)
    [0, 15, 5, 0, 20, 0], # Node 3 (Load)
    [0, 0, 0, 20, 0, 10], # Node 4 (Load)
    [0, 0, 0, 0, 10, 0]   # Node 5 (Load)
])

# Power generation for generators
generation = [50, 40]  # Generator 0 produces 50, Generator 1 produces 40

# Power demand for loads
demand = [10, 10, 25, 15]  # Load 2 requires 10, Load 3 requires 10, etc.


def distribute_power_with_usage(A, generators, loads, generation, demand):
    """
    Distributes power from generators to loads, tracking usage of link capacities.

    Args:
        A: Adjacency matrix representing line capacities.
        generators: List of generator node indices.
        loads: List of load node indices.
        generation: List of generator capacities.
        demand: List of load demands.

    Returns:
        usage: Matrix representing power usage on each link.
        total_power_delivered: Total power successfully delivered to loads.
    """
    num_nodes = len(A)
    usage = np.zeros_like(A)  # Matrix to track power flow
    remaining_gen = generation[:]
    remaining_demand = demand[:]

    # Greedy allocation
    for gen_index, gen_node in enumerate(generators):
        while remaining_gen[gen_index] > 0:
            for load_index, load_node in enumerate(loads):
                if remaining_demand[load_index] <= 0:
                    continue  # Skip fully satisfied loads
                
                # Available capacity on the link
                available_capacity = A[gen_node][load_node] - usage[gen_node][load_node]
                if available_capacity <= 0:
                    continue  # Skip if link is fully utilized
                
                # Power to transfer
                power_to_transfer = min(remaining_gen[gen_index], remaining_demand[load_index], available_capacity)
                usage[gen_node][load_node] += power_to_transfer
                remaining_gen[gen_index] -= power_to_transfer
                remaining_demand[load_index] -= power_to_transfer

    # Calculate total power delivered
    total_power_delivered = sum(d - r for d, r in zip(demand, remaining_demand))

    return usage, total_power_delivered


# Simulate power distribution with usage tracking
usage, total_power_delivered = distribute_power_with_usage(A, generators, loads, generation, demand)

# Display the results
print("Power Flow Matrix (Usage):")
print(usage)

# Total power delivered
print("Total Power Delivered:", total_power_delivered)

# Check satisfaction of demands
for i, load_node in enumerate(loads):
    received_power = usage[:, load_node].sum()
    print(f"Load {load_node} received {received_power} (Demand: {demand[i]})")
