import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d.art3d import Poly3DCollection



# Sample data (replace these with your arrays)
x = np.random.rand(100)  # Sample x coordinates
y = np.random.rand(100)  # Sample y coordinates
z = np.random.rand(100)  # Sample z coordinates


# Combine coordinates into a single array
data = np.column_stack((x, y, z))

# Perform K-means clustering
kmeans = KMeans(n_clusters=3)  # Adjust n_clusters as needed
kmeans.fit(data)
labels = kmeans.labels_
centroids = kmeans.cluster_centers_

# Create 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')



# Plot clusters
for cluster_label in np.unique(labels):
    cluster_mask = labels == cluster_label
    ax.scatter(x[cluster_mask], y[cluster_mask], z[cluster_mask], label=f'Cluster {cluster_label + 1}')


# Plot data
#ax.scatter(x, y, z)


for centroid in centroids:
    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    x_sphere = centroid[0] + 0.1 * np.cos(u) * np.sin(v)
    y_sphere = centroid[1] + 0.1 * np.sin(u) * np.sin(v)
    z_sphere = centroid[2] + 0.1 * np.cos(v)
    ax.plot_surface(x_sphere, y_sphere, z_sphere, color='r', alpha=0.2)


# Set labels
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

plt.savefig('3d_scatter_plot.jpg', format='jpg')
# Show plot
plt.show()
