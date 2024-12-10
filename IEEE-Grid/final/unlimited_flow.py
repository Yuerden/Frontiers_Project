import numpy as np
import random

# Example: Node 0 and 1 are generators, Node 2-5 are loads
# generators = [0, 1]
# generation = [100, 100]  # Generator 0 produces 50, Generator 1 produces 40
# loads = [2, 3, 4, 5, 6]
# demand = [70, 10, 25, 15, 50]  # Load 2 requires 10, Load 3 requires 10, etc.

# adjacency_matrix = np.array([
#     [0, 0, 1, 0, 0, 0, 1],  # Node 0 (Generator)
#     [0, 0, 0, 1, 0, 0, 0],  # Node 1 (Generator)
#     [1, 0, 0, 1, 0, 0, 0],  # Node 2 (Load)
#     [0, 1, 1, 0, 1, 0, 0],  # Node 3 (Load)
#     [0, 0, 0, 1, 0, 1, 0],  # Node 4 (Load)
#     [0, 0, 0, 0, 1, 0, 0],   # Node 5 (Load)
#     [1, 0, 0, 0, 0, 0, 0]   # Node 6 (Load)
# ])


# generators = [0, 1, 2]
# generation = [100, 100, 100]  # Generator 0 produces 50, Generator 1 produces 40
# loads = [3, 4, 5, 6]
# demand = [80, 140, 80, 10]  # Load 2 requires 10, Load 3 requires 10, etc.

# adjacency_matrix = np.array([
#     [0, 0, 0, 1, 1, 1, 0],  # Node 0 (Generator)
#     [0, 0, 0, 0, 0, 1, 1],  # Node 1 (Generator)
#     [0, 0, 0, 1, 1, 0, 0],  # Node 2 (Generator)
#     [0, 0, 0, 0, 0, 0, 0],  # Node 3 (Load)
#     [0, 0, 0, 0, 0, 0, 0],  # Node 4 (Load)
#     [0, 0, 0, 0, 0, 0, 0],  # Node 5 (Load)
#     [0, 0, 0, 0, 0, 0, 0]   # Node 6 (Load)
# ])


import json

with open("systems_config.json", "r") as f:
    loaded_systems = json.load(f)

# Example: access IEEE14 system
ieee14_system = loaded_systems["IEEE14"]
generators = ieee14_system["generators"]
generation = ieee14_system["generation"]
loads = ieee14_system["loads"]
demand = ieee14_system["demand"]
adjacency_matrix = np.array(ieee14_system["adjacency_matrix"])


def find_connected_components(adjacency_matrix):
    """
    Finds connected components in a graph represented by an adjacency matrix.

    Args:
        adjacency_matrix: 2D matrix where 1=link exists, 0=no link.

    Returns:
        List of connected components, where each component is a set of node indices.
    """
    num_nodes = len(adjacency_matrix)
    visited = [False] * num_nodes
    components = []

    def dfs(node, component):
        visited[node] = True
        component.add(node)
        for neighbor, is_connected in enumerate(adjacency_matrix[node]):
            if is_connected and not visited[neighbor]:
                dfs(neighbor, component)

    for node in range(num_nodes):
        if not visited[node]:
            component = set()
            dfs(node, component)
            components.append(component)

    return components

def load_demand_by_component(adjacency_matrix, generators, loads, demand):
    """
    Finds connected components among loads and calculates their total demand.

    Args:
        adjacency_matrix: Full adjacency matrix (1=link exists, 0=no link).
        generators: List of generator node indices.
        loads: List of load node indices.
        demand: List of load demands.

    Returns:
        components: A list of connected components, each as a set of load indices.
        total_demand_per_component: A list of total power demands for each connected component.
    """
    # Extract the subgraph of loads
    load_A = np.zeros((len(loads), len(loads)), dtype=int)
    for i, node in enumerate(range(len(generators), len(adjacency_matrix))):
        for j, neighbor in enumerate(range(len(generators), len(adjacency_matrix))):
            load_A[i][j] = adjacency_matrix[node][neighbor]

    # Find connected components in the load subgraph
    components = find_connected_components(load_A)

    # Map components back to global indices
    global_components = []
    for component in components:
        global_component = {loads[node] for node in component}  # Map local indices to global indices
        global_components.append(global_component)

    # Calculate the total demand for each component
    total_demand_per_component = []
    for component in global_components:
        total_demand = sum(demand[loads.index(load)] for load in component)
        total_demand_per_component.append(total_demand)

    return global_components, total_demand_per_component

def see_component_stats(adjacency_matrix, generators, loads, demand):
    components, total_demand_per_component = load_demand_by_component(adjacency_matrix, generators, loads, demand)

    # Print results
    for i, (component, total_demand) in enumerate(zip(components, total_demand_per_component)):
        print(f"Component {i + 1}: Nodes {component}, Total Demand = {total_demand}")

def create_bipartite_graph(adjacency_matrix, generators, loads, demand):
    """
    Reduces the graph into a bipartite graph of generators and load components.

    Args:
        adjacency_matrix: Full adjacency matrix (1=link exists, 0=no link).
        generators: List of generator node indices.
        loads: List of load node indices.
        demand: List of load demands.

    Returns:
        bipartite_graph: Dictionary representing the bipartite graph.
                         Keys are generators, values are lists of connected load components.
        component_demands: List of total demands for each load component.
    """
    # Step 1: Identify connected components of loads
    components, total_demand_per_component = load_demand_by_component(adjacency_matrix, generators, loads, demand)

    # Step 2: Create bipartite graph
    bipartite_graph = {gen: [] for gen in generators}
    component_demands = []

    for comp_index, component in enumerate(components):
        component_demands.append(total_demand_per_component[comp_index])
        for gen in generators:
            # Check if the generator is connected to any load in the component
            for load in component:
                if adjacency_matrix[gen][load] == 1:
                    bipartite_graph[gen].append(comp_index)
                    break  # No need to check other loads in this component

    return bipartite_graph, component_demands

def optimize_power_distribution_refined(bipartite_graph, component_demands, generation):
    """
    Optimizes power distribution, prioritizing components with the least number of generators
    and generators with the least number of connections.

    Args:
        bipartite_graph: Dictionary representing the bipartite graph.
                         Keys are generators, values are lists of connected components.
        component_demands: List of total demands for each load component.
        generation: List of generator capacities.

    Returns:
        allocation: 2D list where allocation[i][j] is the power allocated from generator i to component j.
        total_power_delivered: Total power successfully delivered to all components.
    """
    num_generators = len(generation)
    num_components = len(component_demands)

    # Initialize allocation matrix
    allocation = [[0] * num_components for _ in range(num_generators)]

    # Remaining capacity of generators and demands of components
    remaining_gen = generation[:]
    remaining_demand = component_demands[:]

    # Step 1: Calculate generator availability for each component
    component_to_generators = {comp: [] for comp in range(num_components)}
    generator_to_components = {gen: [] for gen in range(num_generators)}

    for gen_index, gen_node in enumerate(bipartite_graph.keys()):
        for comp_index in bipartite_graph[gen_node]:
            component_to_generators[comp_index].append(gen_index)
            generator_to_components[gen_index].append(comp_index)

    # Step 2: Sort components by number of connected generators (ascending order)
    sorted_components = sorted(component_to_generators.keys(), key=lambda c: len(component_to_generators[c]))

    # Step 3: Allocate power prioritizing generator connections
    for comp_index in sorted_components:
        # Sort generators connected to this component by the number of other components they serve (ascending order)
        sorted_generators = sorted(component_to_generators[comp_index],
                                   key=lambda g: len(generator_to_components[g]))

        for gen_index in sorted_generators:
            if remaining_gen[gen_index] <= 0 or remaining_demand[comp_index] <= 0:
                continue

            # Allocate power
            power_to_transfer = min(remaining_gen[gen_index], remaining_demand[comp_index])
            allocation[gen_index][comp_index] += power_to_transfer
            remaining_gen[gen_index] -= power_to_transfer
            remaining_demand[comp_index] -= power_to_transfer

    # Calculate total power delivered
    total_power_delivered = sum(
        component_demands[i] - remaining_demand[i] for i in range(num_components)
    )

    return allocation, total_power_delivered

def system_state(adjacency_matrix, generators, generation, loads, demand):
    bipartite_graph, component_demands = create_bipartite_graph(adjacency_matrix, generators, loads, demand)

    print("Bipartite Graph:")
    for gen, comps in bipartite_graph.items():
        print(f"Generator {gen} -> Components {comps}")

    print("\nComponent Demands:")
    for i, comp_dem in enumerate(component_demands):  # Renamed 'demand' to 'comp_dem'
        print(f"Component {i}: Total Demand = {comp_dem}")

    # Optimize power distribution
    allocation, total_power_delivered = optimize_power_distribution_refined(
        bipartite_graph, component_demands, generation
    )

    print("Power Allocation:")
    for i, row in enumerate(allocation):
        print(f"Generator {i}: {row}")

    print(f"\nTotal Power Delivered: {total_power_delivered}")

    print("\nRemaining Demands for Each Component:")
    for i, comp_dem in enumerate(component_demands):
        print(f"Component {i}: Remaining Demand = {comp_dem - sum(row[i] for row in allocation)}")

    # Recompute 'components' since we need them for load states
    # (We know create_bipartite_graph calls load_demand_by_component, so we can re-run that function)
    components, _ = load_demand_by_component(adjacency_matrix, generators, loads, demand)

    # Step 1: Calculate component received power
    component_received_power = [sum(row[comp] for row in allocation) for comp in range(len(component_demands))]

    # Step 2: Calculate load states using 'components' (not bipartite_graph.values())
    load_states = {}
    for comp_index, comp_load_set in enumerate(components):
        received_power_ratio = component_received_power[comp_index] / component_demands[comp_index] if component_demands[comp_index] > 0 else 0
        for load in comp_load_set:
            # 'load' is a load node, map it to 'demand'
            load_desire = demand[loads.index(load)]
            load_states[load] = load_desire * received_power_ratio

    # Step 3: Calculate x_eff
    x_eff_numerator = 0
    x_eff_denominator = len(loads)
    for load_node in loads:
        state = load_states.get(load_node, 0)
        outgoing_degree = sum(adjacency_matrix[load_node])
        x_eff_numerator += state * outgoing_degree

    x_eff = x_eff_numerator / x_eff_denominator if x_eff_denominator > 0 else 0
    print(f"\nSystem Efficiency (x_eff): {x_eff}")
    return x_eff

def simulate_line_failure(f_l, adjacency_matrix, generators, generation, loads, demand):
    # Convert adjacency_matrix to a NumPy array if it isn't already
    A = np.array(adjacency_matrix)
    
    # Identify all edges (nonzero entries)
    edges = [(i, j) for i in range(len(A)) for j in range(len(A)) if A[i][j] > 0]

    if not edges:
        print("No lines present to remove.")
        return system_state(A, generators, loads, demand)

    # Determine how many lines to remove
    total_lines = len(edges)
    lines_to_remove = int(f_l * total_lines)

    # Randomly choose lines to remove
    lines_removed = random.sample(edges, lines_to_remove)

    # Create a copy to avoid mutating the original matrix
    new_A = A.copy()
    for (i, j) in lines_removed:
        new_A[i][j] = 0

    # Run the system state with the new adjacency matrix
    system_state(new_A, generators, generation, loads, demand)

def simulate_demand_increase(f_d, adjacency_matrix, generators, generation, loads, demand):
    # f_d: fraction increase in demand
    # Increase each load's demand by (f_d * original_demand)
    new_demand = [d * (1 + f_d) for d in demand]
    system_state(adjacency_matrix, generators, new_demand, loads, new_demand)

def simulate_generation_decrease(f_g, adjacency_matrix, generators, generation, loads, demand):
    # f_g: fraction decrease in generation
    # Decrease each generator's capacity by f_g of its original capacity
    new_generation = [g * (1 - f_g) for g in generation]
    system_state(adjacency_matrix, generators, new_generation, loads, demand)

def simulate_generator_failure(f_g, adjacency_matrix, generators, generation, loads, demand):
    # f_g: fraction of generators to fail
    # Randomly pick f_g * len(generators) generators and set their generation to zero
    num_failures = int(f_g * len(generators))
    failed_gens = random.sample(generators, num_failures)

    new_generation = generation[:]
    for gen in failed_gens:
        gen_index = generators.index(gen)
        new_generation[gen_index] = 0

    system_state(adjacency_matrix, generators, new_generation, loads, demand)

def simulate_load_failure(f_l, adjacency_matrix, generators, generation, loads, demand):
    # f_l: fraction of loads to fail
    # Randomly pick f_l * len(loads) loads and set their demand to zero
    num_failures = int(f_l * len(loads))
    failed_loads = random.sample(loads, num_failures)

    new_demand = demand[:]
    for ld in failed_loads:
        ld_index = loads.index(ld)
        new_demand[ld_index] = 0

    system_state(adjacency_matrix, generators, generation, loads, new_demand)

system_state(adjacency_matrix, generators, generation, loads, demand)
simulate_line_failure(0.5, adjacency_matrix, generators, generation, loads, demand)
