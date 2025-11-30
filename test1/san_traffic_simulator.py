import pandas as pd
import numpy as np
import random

print("Starting SAN traffic simulation...")

# Simulation parameters
TIMESTEPS = 2000  # Number of simulated minutes
NORMAL_IOPS_MEAN = 300
NORMAL_IOPS_STD = 100
NORMAL_BW_MEAN = 80
NORMAL_BW_STD = 20
LATENCY_BASE = 2
LATENCY_NOISE = 1

# Congestion spike parameters
SPIKE_PROBABILITY = 0.05
SPIKE_DURATION = 30  # in timesteps
SPIKE_IOPS_MEAN = 1200
SPIKE_IOPS_STD = 200
SPIKE_BW_MEAN = 250
SPIKE_BW_STD = 40

data = []
in_spike = 0

for i in range(TIMESTEPS):
    # Decide if a spike starts
    if in_spike == 0 and random.random() < SPIKE_PROBABILITY:
        in_spike = SPIKE_DURATION
        print(f"Injecting congestion spike at timestep {i} for {SPIKE_DURATION} steps.")

    if in_spike > 0:
        # Generate spike data
        read_iops = int(np.random.normal(SPIKE_IOPS_MEAN * 0.6, SPIKE_IOPS_STD))
        write_iops = int(np.random.normal(SPIKE_IOPS_MEAN * 0.4, SPIKE_IOPS_STD))
        bw_usage = int(np.random.normal(SPIKE_BW_MEAN, SPIKE_BW_STD))
        queue_length = int(np.random.uniform(50, 100) + (in_spike / SPIKE_DURATION) * 50)
        
        # Latency is highly correlated with queue and IOPS during a spike
        latency = (LATENCY_BASE * 5) + (queue_length / 10) + (read_iops + write_iops) / 100 + np.random.normal(0, LATENCY_NOISE * 2)
        
        congestion = 1  # Mark as a congestion event
        in_spike -= 1
    
    else:
        # Generate normal data
        read_iops = int(np.random.normal(NORMAL_IOPS_MEAN * 0.7, NORMAL_IOPS_STD))
        write_iops = int(np.random.normal(NORMAL_IOPS_MEAN * 0.3, NORMAL_IOPS_STD))
        bw_usage = int(np.random.normal(NORMAL_BW_MEAN, NORMAL_BW_STD))
        queue_length = int(np.random.uniform(5, 20))
        
        # Normal latency
        latency = LATENCY_BASE + (queue_length / 10) + (read_iops + write_iops) / 400 + np.random.normal(0, LATENCY_NOISE)

        # Use the plan's threshold for labeling
        if latency > 10:
             congestion = 1
        else:
             congestion = 0

    # Ensure no negative values
    read_iops = max(50, read_iops)
    write_iops = max(50, write_iops)
    bw_usage = max(10, bw_usage)
    latency = max(1, round(latency, 2))

    data.append([
        i,
        read_iops,
        write_iops,
        queue_length,
        bw_usage,
        latency,
        congestion
    ])

# Create DataFrame
columns = ['Time', 'Read_IOPS', 'Write_IOPS', 'Queue_Length', 'Bandwidth_MBps', 'Latency_ms', 'Congestion']
df = pd.DataFrame(data, columns=columns)

# Save to CSV
df.to_csv('san_traffic.csv', index=False)

print(f"\nSimulation complete. Generated {TIMESTEPS} records.")
print(f"Total congestion events labeled: {df['Congestion'].sum()}")
print("Data saved to 'san_traffic.csv'.")
