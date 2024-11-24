import pandapower as pp
import pandapower.networks as pn
import pandapower.topology as top
import pandapower.plotting as plot
import networkx as nx
import matplotlib.pyplot as plt
import random

# Load the IEEE 14-bus test network
net = pn.case14()


# View buses
print(net.bus)

# View lines
print(net.line)

# View loads
print(net.load)

# View generators
print(net.gen)

# View external grids (slack buses)
print(net.ext_grid)


# Simple plot of the network
plot.simple_plot(net)
plt.show()


# Run a power flow calculation
pp.runpp(net)

# Check if the power flow converged
print(f"Power flow converged: {net.converged}")


def simulate_failures(net, failure_probability):
    # Copy the network to avoid modifying the original
    net = net.deepcopy()
    
    # Simulate line failures
    for line_idx in net.line.index:
        if random.random() < failure_probability:
            net.line.in_service.at[line_idx] = False
    
    # Simulate bus failures (optional)
    # for bus_idx in net.bus.index:
    #     if random.random() < failure_probability:
    #         net.bus.in_service.at[bus_idx] = False

    return net


def analyze_network(net):
    # Try running power flow
    try:
        pp.runpp(net)
        converged = net.converged
    except:
        converged = False

    # Create the NetworkX graph
    G = top.create_nxgraph(net, include_lines=True, include_trafos=False)

    # Analyze connectivity
    if len(G) == 0:
        largest_cc_size = 0
        is_connected = False
    else:
        largest_cc = max(nx.connected_components(G), key=len)
        largest_cc_size = len(largest_cc)
        is_connected = nx.is_connected(G)

    return {
        'converged': converged,
        'largest_cc_size': largest_cc_size,
        'is_connected': is_connected,
        'graph': G
    }


import numpy as np

failure_probabilities = np.linspace(0, 0.5, 6)  # From 0% to 50% in 10% increments
results = []

for fp in failure_probabilities:
    print(f"Simulating failures with probability: {fp}")
    
    # Simulate failures
    failed_net = simulate_failures(net, fp)
    
    # Analyze the network
    analysis = analyze_network(failed_net)
    analysis['failure_probability'] = fp
    results.append(analysis)


# Extract data for plotting
fps = [res['failure_probability'] for res in results]
cc_sizes = [res['largest_cc_size'] for res in results]

# Plotting
plt.figure(figsize=(8, 6))
plt.plot(fps, cc_sizes, marker='o')
plt.xlabel('Failure Probability')
plt.ylabel('Size of Largest Connected Component')
plt.title('Network Resilience Analysis of IEEE 14-Bus System')
plt.grid(True)
plt.show()


# Visualize the network after failures
failed_net = simulate_failures(net, 0.3)  # Example with 30% failure probability
analysis = analyze_network(failed_net)

# Plot failed network
plot.simple_plot(failed_net)
plt.show()


num_runs = 10
for fp in failure_probabilities:
    cc_sizes = []
    for _ in range(num_runs):
        failed_net = simulate_failures(net, fp)
        analysis = analyze_network(failed_net)
        cc_sizes.append(analysis['largest_cc_size'])
    avg_cc_size = sum(cc_sizes) / num_runs
    # Store or plot the average cc_size
