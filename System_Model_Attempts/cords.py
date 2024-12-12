import pandapower as pp
import pandapower.networks as pn
import pandapower.topology as top
import pandapower.plotting as plot
import networkx as nx
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the IEEE 14-bus test network
net = pn.case30()

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

# Get bus indices
bus_indices = net.bus.index.tolist()

# Define coordinate ranges
lat_range = (30.0, 31.0)
lon_range = (-100.0, -99.0)

# Assign random coordinates to buses
np.random.seed(42)  # For reproducibility
net.bus['lat'] = np.random.uniform(lat_range[0], lat_range[1], size=len(bus_indices))
net.bus['lon'] = np.random.uniform(lon_range[0], lon_range[1], size=len(bus_indices))

# Analyze the network
analysis = analyze_network(net)
print(f"Power flow converged: {analysis['converged']}")
print(f"Size of the largest connected component: {analysis['largest_cc_size']}")
print(f"Is network connected: {analysis['is_connected']}")

plt.figure(figsize=(8, 6))
plt.scatter(net.bus['lon'], net.bus['lat'], c='blue', s=100)
for idx, row in net.bus.iterrows():
    plt.text(row['lon'], row['lat'], str(idx), fontsize=12, ha='right')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Bus Locations in the IEEE 14-Bus System')
plt.grid(True)
plt.show()

# Simple plot of the network
plot.simple_plot(net)
plt.show()


def simulate_weather_event(net, weather_center, weather_radius, high_failure_prob=0.5, low_failure_prob=0.01):
    """
    Simulate weather events by increasing failure probabilities of buses and lines within a certain radius.
    """
    # Copy the network to avoid modifying the original
    net = net.deepcopy()

    # Calculate distances of buses from the weather event center
    distances = np.sqrt((net.bus['lat'] - weather_center[0])**2 + (net.bus['lon'] - weather_center[1])**2)

    # Identify affected buses
    affected_buses = net.bus.index[distances <= weather_radius].tolist()

    # Set failure probabilities
    net.bus['failure_prob'] = low_failure_prob
    net.bus.loc[affected_buses, 'failure_prob'] = high_failure_prob

    # Simulate bus failures
    for idx in net.bus.index:
        if random.random() < net.bus.at[idx, 'failure_prob']:
            net.bus.at[idx, 'in_service'] = False

    # Identify lines connected to affected buses
    affected_lines = net.line[
        net.line['from_bus'].isin(affected_buses) | net.line['to_bus'].isin(affected_buses)
    ].index.tolist()

    # Set failure probabilities for lines
    net.line['failure_prob'] = low_failure_prob
    net.line.loc[affected_lines, 'failure_prob'] = high_failure_prob

    # Simulate line failures
    for idx in net.line.index:
        if random.random() < net.line.at[idx, 'failure_prob']:
            net.line.at[idx, 'in_service'] = False

    return net, affected_buses, affected_lines


# Define the weather event center and radius
weather_center = (30.5, -99.5)
weather_radius = 0.3  # Adjust as needed

# Simulate weather event
failed_net, affected_buses, affected_lines = simulate_weather_event(
    net,
    weather_center,
    weather_radius,
    high_failure_prob=0.5,
    low_failure_prob=0.5
)


# Analyze the network
analysis = analyze_network(failed_net)
print(f"Power flow converged: {analysis['converged']}")
print(f"Size of the largest connected component: {analysis['largest_cc_size']}")
print(f"Is network connected: {analysis['is_connected']}")

# Visualize affected buses
plt.figure(figsize=(8, 6))
plt.scatter(net.bus['lon'], net.bus['lat'], c='blue', s=100, label='Unaffected Buses')
plt.scatter(net.bus.loc[affected_buses, 'lon'], net.bus.loc[affected_buses, 'lat'], c='red', s=100, label='Affected Buses')
circle = plt.Circle((weather_center[1], weather_center[0]), weather_radius, color='green', fill=False, linestyle='--', label='Weather Event')
plt.gca().add_artist(circle)
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Weather Event Impact on Buses')
plt.legend()
plt.grid(True)
plt.show()

# Simple plot of the network
plot.simple_plot(net)
plt.show()