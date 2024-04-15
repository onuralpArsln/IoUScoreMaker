import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# Sample data (replace these with your arrays)
x = np.random.rand(100)  # Sample x coordinates
y = np.random.rand(100)  # Sample y coordinates
z = np.random.rand(100)  # Sample z coordinates

# Create 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot data
ax.scatter(x, y, z)

# Set labels
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# Show plot
plt.show()
