import json
import math
import matplotlib.pyplot as plt

# Paths to data files
data_path = {
    0: './data/export/walk_real.json',
    1: './data/export/visual_walk_6136.json'
}
data_path_index = 0  # Select which data file to use

# Sensor types available in the data
sensor_types = {
    0: 'accelerometer',
    1: 'gyroscope',
    2: 'relOrientation'
}
sensor_index = 1  # Select which sensor type to plot

# Load the selected JSON data file
with open(data_path[data_path_index], 'r') as f:
    data = json.load(f)

# Extract sensor data
sensor_data = data['sensorData']
accel = sensor_data[sensor_types[sensor_index]]

# Extract time and axis values
t = [entry['t'] for entry in accel]
x = [entry['x'] for entry in accel]
y = [entry['y'] for entry in accel]
z = [entry['z'] for entry in accel]

# Define time window for plotting (in ms)
start_time = 5000
end_time = 10000

# Find indices corresponding to the time window
start_index = next(i for i, entry in enumerate(accel) if entry['t'] >= start_time)
end_index = next(i for i, entry in enumerate(accel) if entry['t'] > end_time)

# Slice data to the selected time window
t = t[start_index:end_index]
x = x[start_index:end_index]
y = y[start_index:end_index]
z = z[start_index:end_index]

# Compute vector magnitude at each time step
magnitude = [math.sqrt(xi**2 + yi**2 + zi**2) for xi, yi, zi in zip(x, y, z)]

# Normalize time to start from zero
t = [entry - start_time for entry in t]

# Plotting
plt.figure(figsize=(5, 5))
if sensor_types[sensor_index] == 'accelerometer':
    plt.plot(t, x, label='x')
    plt.plot(t, y, label='y')
    plt.plot(t, z, label='z')
elif sensor_types[sensor_index] == 'gyroscope':
    plt.plot(t, x, label='$\omega_x$')
    plt.plot(t, y, label='$\omega_y$')
    plt.plot(t, z, label='$\omega_z$')
elif sensor_types[sensor_index] == 'relOrientation':
    plt.plot(t, x, label='Roll ($\phi$)')
    plt.plot(t, y, label='Pitch ($\\theta$)')
    plt.plot(t, z, label='Yaw ($\psi$)')

# Plot vector magnitude
plt.plot(t, magnitude, label='Vector size', linestyle='--', color='red')

plt.xlabel('Time (ms)')

# Set y-axis label based on sensor type
if sensor_types[sensor_index] == 'accelerometer':
    plt.ylabel('Acceleration [$m/s^2$]')
elif sensor_types[sensor_index] == 'gyroscope':
    plt.ylabel('Angular Velocity [$rad/s$]')
elif sensor_types[sensor_index] == 'relOrientation':
    plt.ylabel('Euler Angle [$rad$]')

plt.title('Real Data')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()